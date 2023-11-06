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


import copy
import csv
from datetime import datetime
from typing import Optional

from dynast.predictors.dynamic_predictor import Predictor
from dynast.search.evaluation_interface import EvaluationInterface
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


class EvaluationInterfaceViTQuantized(EvaluationInterface):
    def __init__(
        self,
        evaluator,
        manager,
        optimization_metrics: list = ['accuracy_top1', 'latency'],
        measurements: list = ['accuracy_top1', 'latency'],
        csv_path=None,
        predictor_mode: bool = False,
    ):
        super().__init__(evaluator, manager, optimization_metrics, measurements, csv_path, predictor_mode)

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        sample = {
            'vit_hidden_sizes': 768,
            'num_layers': param_dict['num_layers'][0],
            'num_attention_heads': param_dict['num_attention_heads'],
            'vit_intermediate_sizes': param_dict["vit_intermediate_sizes"],
        }

        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'macs', 'accuracy_top1']:
            individual_results[metric] = 0

        # Predictor Mode
        if self.predictor_mode == True:
            if 'params' in self.optimization_metrics:
                individual_results['params'] = self.evaluator.estimate_parameters(
                    self.manager.onehot_custom(param_dict, max_layers=self.evaluator.max_layers).reshape(1, -1)
                )[0]
            if 'latency' in self.optimization_metrics:
                individual_results['latency'] = self.evaluator.estimate_latency(
                    self.manager.onehot_custom(param_dict, max_layers=self.evaluator.max_layers).reshape(1, -1)
                )[0]
            if 'macs' in self.optimization_metrics:
                individual_results['macs'] = self.evaluator.estimate_macs(
                    self.manager.onehot_custom(param_dict, max_layers=self.evaluator.max_layers).reshape(1, -1)
                )[0]
            if 'accuracy_top1' in self.optimization_metrics:
                individual_results['accuracy_top1'] = self.evaluator.estimate_accuracy_imagenet(
                    self.manager.onehot_custom(param_dict, max_layers=self.evaluator.max_layers).reshape(1, -1)
                )[0]

        # Validation Mode
        else:
            if 'macs' in self.measurements or 'params' in self.measurements:
                individual_results['macs'], individual_results['params'] = self.evaluator.validate_macs(subnet_sample)
            if 'latency' in self.measurements:
                individual_results['latency'], _ = self.evaluator.measure_latency(subnet_sample)
            if 'accuracy_top1' in self.measurements:
                individual_results['accuracy_top1'] = self.evaluator.validate_accuracy_imagenet(subnet_sample)

        subnet_sample = param_dict
        sample = param_dict
        # Write result for csv_path
        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [
                    subnet_sample,
                    date,
                    individual_results['params'],
                    individual_results['latency'],
                    individual_results['macs'],
                    individual_results['accuracy_top1'],
                ]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        individual_results['accuracy_top1'] = -individual_results['accuracy_top1']
        # Return results to pymoo
        if len(self.optimization_metrics) == 1:
            return sample, individual_results[self.optimization_metrics[0]]
        elif len(self.optimization_metrics) == 2:
            return (
                sample,
                individual_results[self.optimization_metrics[0]],
                individual_results[self.optimization_metrics[1]],
            )
        elif len(self.optimization_metrics) == 3:
            return (
                sample,
                individual_results[self.optimization_metrics[0]],
                individual_results[self.optimization_metrics[1]],
                individual_results[self.optimization_metrics[2]],
            )
        else:
            log.error('Number of optimization_metrics is out of range. 1-3 supported.')
            return None
