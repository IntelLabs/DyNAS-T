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


from typing import Dict, Union

from dynast.predictors.dynamic_predictor import Predictor


class Runner(object):
    def __init__(
        self,
        supernet: str,
        dataset_path: Union[None, str] = None,
        predictors: Dict[str, Predictor] = {},
        batch_size: int = 128,
        eval_batch_size: int = 128,
        dataloader_workers: int = 4,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.supernet = supernet
        self.predictors = predictors
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.test_fraction = test_fraction
        self.dataset_path = dataset_path
        self.dataloader_workers = dataloader_workers
        self.verbose = verbose

    def estimate_metric(self, metric: str, subnet_cfg) -> float:
        predictor: Union[Predictor, None] = self.predictors.get(metric)
        if predictor is None:
            raise Exception(f'No predictor for metric {metric} was found.')

        predicted_val = predictor.predict(subnet_cfg)

        return predicted_val
