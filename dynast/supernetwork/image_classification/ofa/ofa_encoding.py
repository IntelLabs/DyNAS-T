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

from typing import Dict, List, Tuple, Union

import numpy as np

from dynast.search.encoding import EncodingBase


class OFAMobileNetV3Encoding(EncodingBase):
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):
        super().__init__(param_dict, verbose, seed)

    def construct_maps(self, keys: Union[List, Tuple]) -> Dict[int, int]:
        d: Dict[int, int] = dict()
        keys = list(set(keys))
        for k in keys:
            if k not in d:
                d[k] = len(list(d.keys()))
        return d

    def onehot_custom(self, ks_list: list, ex_list: list, d_list: list) -> np.ndarray:

        ks_map = self.construct_maps(keys=(3, 5, 7))
        ex_map = self.construct_maps(keys=(3, 4, 6))
        dp_map = self.construct_maps(keys=(2, 3, 4))

        # This function converts a network config to a feature vector (128-D).
        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

        # convert to onehot
        ks_onehot = [0 for _ in range(60)]
        ex_onehot = [0 for _ in range(60)]

        for i in range(20):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + ex_map[ex_list[i]]] = 1

        return np.array(ks_onehot + ex_onehot)


class OFAResNet50Encoding(EncodingBase):
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):
        super().__init__(param_dict, verbose, seed)
