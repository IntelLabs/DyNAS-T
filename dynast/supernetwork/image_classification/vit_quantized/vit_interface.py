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


from typing import Optional

from dynast.predictors.dynamic_predictor import Predictor
from dynast.supernetwork.image_classification.ofa_quantization.quantization_interface import Quantization
from dynast.supernetwork.image_classification.vit.vit_interface import ViTRunner, load_supernet
from dynast.utils import log
from dynast.utils.datasets import ImageNet


class ViTQuantizedRunner(ViTRunner):
    def __init__(
        self,
        supernet,
        dataset_path,
        acc_predictor: Optional[Predictor] = None,
        macs_predictor: Optional[Predictor] = None,
        latency_predictor: Optional[Predictor] = None,
        params_predictor: Optional[Predictor] = None,
        batch_size: int = 128,
        eval_batch_size: int = 128,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        dataloader_workers: int = 4,
        test_fraction: float = 1.0,
        warmup_steps: int = 10,
        measure_steps: int = 100,
        mp_calibration_samples: int = 100,
    ) -> None:
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.latency_predictor = latency_predictor
        self.params_predictor = params_predictor
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.test_fraction = test_fraction
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.mp_calibration_samples = mp_calibration_samples
        self.dataloader_workers = dataloader_workers

        self.supernet_model, self.max_layers = load_supernet(self.checkpoint_path)

        self._init_data()

        self.quantizer = Quantization(
            calibration_dataloader=self.calibration_dataloader, mp_calibration_samples=self.mp_calibration_samples
        )

    def _init_data(self) -> None:
        ImageNet.PATH = self.dataset_path
        if self.dataset_path:
            self.dataloader = ImageNet.validation_dataloader(
                batch_size=self.eval_batch_size,
                num_workers=self.dataloader_workers,
                fraction=self.test_fraction,
            )
            # TODO(macsz) Consider adding `train_batch_size`
            self.calibration_dataloader = ImageNet.train_dataloader(batch_size=self.eval_batch_size)

        else:
            self.dataloader = None
            log.warning('No dataset path provided. Cannot validate sub-networks.')
