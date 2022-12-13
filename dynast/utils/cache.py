# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any, Union


class Cache(object):
    """Simple caching mechanism to speed-up experiments by re-using evaluated configurations
    and storing atomic results in separate files.

    Example:
    ```
    # `x` - (list) one-hot encoded configuration
    cache_key = Cache.key(x)

    if not self.cache.exists(cache_key):
        top1, top5, gflops, model_params = self.evaluator.validate_subnet(subnet_sample)
        latency = self.evaluator.benchmark_subnet()
        date = str(datetime.now())
        self.cache.update(key=cache_key, payload=[subnet_sample, date, float(latency), float(top1)])
    else:
        subnet_sample, date, latency, top1 = self.cache.get(key=cache_key)
    ```
    """

    _hits = 0

    def __init__(self, name: str, cache_dir: str = '/store/.torch/dynast_cache') -> None:
        """
        Params:
        - `name` - (str) unique name of your experiment. Cache name should be unique
            for each platform and experiment configuration.
        - `cache_dir` - (str, optional) root directory where cache will be stored.
        """
        self.name = name
        self.cache_dir = cache_dir
        os.makedirs(os.path.join(self.cache_dir, self.name), exist_ok=True)

    def update(self, key: str, payload: Any) -> None:
        """Update (add new or replace existing) cache stored under `key`.

        Params:
        - `key` - (str) key under which `payload` will be stored
        - `payload` - (Any) object to be cached. The only requirement is for `payload`
            to be representable in the JSON format.
        """
        with open(self._get_key_path(key), 'w') as f:
            return json.dump(payload, f, indent=4)

    def get(self, key: str) -> Union[Any, None]:
        """Retrieve cache stored under `key`. Returns `None` if 'key` doesnt exist.

        Params:
        - `key` - (str) `key` to retrieve payload.
        """
        try:
            with open(self._get_key_path(key), 'r') as f:
                self._hits += 1
                return json.load(f)
        except FileNotFoundError:
            return None

    @staticmethod
    def key(o: list) -> str:
        """Translate PyMoo vector of one-hot encoded configuration into a file-name friendly string.

        Params:
        `o` - (list) Network's one-hot configuration
        """
        return '_'.join([str(_) for _ in o])

    def exists(self, key: str) -> bool:
        """Checks if cache `key` exists.

        Params:
        - `key` - (str) `key` to retrieve payload.
        """
        return os.path.exists(self._get_key_path(key))

    def _get_key_path(self, key: str) -> str:
        """
        Params:
        - `key` - (str) `key` under which the payload is/will be stored.
        """
        return os.path.join(self.cache_dir, self.name, '{}.json'.format(key))
