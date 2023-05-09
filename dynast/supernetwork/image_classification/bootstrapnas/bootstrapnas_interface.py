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
from typing import Tuple

from nncf.experimental.torch.nas.bootstrapNAS.search.supernet import SuperNetwork  # TODO(macsz) Lazy import
import numpy as np
import torch

from dynast.measure.latency import auto_steps
from dynast.predictors.dynamic_predictor import Predictor
from dynast.search.evaluation_interface import EvaluationInterface
from dynast.utils import log
from dynast.utils.datasets import CIFAR10
from dynast.utils.nn import get_macs, measure_latency, validate_classification


class BootstrapNASRunner:
    """The BootstrapNASRunner class manages the sub-network selection from the BootstrapNAS super-network and
    the validation measurements of the sub-networks.
    """

    def __init__(
        self,
        bootstrapnas: SuperNetwork,
        supernet: str,
        dataset_path: str = None,
        acc_predictor: Predictor = None,
        macs_predictor: Predictor = None,
        latency_predictor: Predictor = None,
        params_predictor: Predictor = None,
        batch_size: int = 1,
        device: str = 'cpu',
    ):
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.latency_predictor = latency_predictor
        self.params_predictor = params_predictor
        self.batch_size = batch_size
        self.device = device
        self.bootstrapnas = bootstrapnas
        self.dataset_path = dataset_path

    def estimate_accuracy_top1(self, subnet_cfg):
        top1 = self.acc_predictor.predict(subnet_cfg)
        return top1

    def estimate_macs(self, subnet_cfg: np.ndarray) -> int:
        macs = self.macs_predictor.predict(subnet_cfg)
        return macs

    def estimate_latency(self, subnet_cfg):
        latency = self.latency_predictor.predict(subnet_cfg)
        return latency

    def estimate_parameters(self, subnet_cfg):
        parameters = self.params_predictor.predict(subnet_cfg)
        return parameters

    def validate_top1(self, pymoo_vector, device=None):
        if device is None:
            device = self.device

        CIFAR10.PATH = self.dataset_path

        validation_dataloader = CIFAR10.validation_dataloader(
            batch_size=self.batch_size
        )  # TODO(macsz) Move to constructor so it's not initialized from scratch every time.

        model = self._get_subnet(pymoo_vector, device)

        losses, top1, top5 = validate_classification(
            model=model,
            data_loader=validation_dataloader,
            device=device,
        )

        return top1

    def validate_macs_params(self, pymoo_vector: dict, device: str = None) -> float:
        """Measure Torch model's FLOPs/MACs as per FVCore calculation
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            `macs`
        """
        if device is None:
            device = self.device

        model = self._get_subnet(pymoo_vector, device)

        params = sum(param.numel() for param in model.parameters())

        macs = get_macs(
            model=model,
            input_size=(1, 3, 32, 32),  # batch size does not matter for MACs (scales linearly).
            device=device,
        )

        return macs, params

    @torch.no_grad()
    def measure_latency(
        self,
        pymoo_vector: dict,
        input_size: tuple = (1, 3, 32, 32),
        warmup_steps: int = 10,
        measure_steps: int = 50,
        device: str = None,
    ) -> Tuple[float, float]:
        """Measure Torch model's latency.
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            mean latency; std latency
        """
        # TODO(macsz) this function can be replaced with the one in `dynast.utils`.
        # Per Daniel's point, we should also consider settubg `omp_num_threads` here,
        if device is None:
            device = self.device

        model = self._get_subnet(pymoo_vector, device)

        if not warmup_steps:
            warmup_steps = auto_steps(input_size[0], is_warmup=True)
        if not measure_steps:
            measure_steps = auto_steps(input_size[0])

        latency_mean, latency_std = measure_latency(
            model=model,
            input_size=input_size,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            device=device,
            ignore_batchnorm=False,
        )
        return latency_mean, latency_std

    def _get_subnet(self, pymoo_vector, device):
        subnet_sample = copy.deepcopy(pymoo_vector)

        self.bootstrapnas.activate_config(self.bootstrapnas.get_config_from_pymoo(subnet_sample))
        model = self.bootstrapnas.get_active_subnet()
        model = model.to(device)
        return model


class EvaluationInterfaceBootstrapNAS(EvaluationInterface):
    def __init__(
        self,
        evaluator,
        manager,
        optimization_metrics: list = ['accuracy_top1', 'macs'],
        measurements: list = ['accuracy_top1', 'macs'],
        csv_path=None,
        predictor_mode: bool = False,
    ):
        super().__init__(evaluator, manager, optimization_metrics, measurements, csv_path, predictor_mode)

    def eval_subnet(self, x):
        """This handles the evaluation (prediction or validation) for various sub-network configurations

        Params:
        - x: pymoo numpy vector
        """
        np_x = np.array(x)
        x = self.manager.reconstruct_pymoo_vector(x)

        # PyMoo vector to Super-Network Parameter Mapping, BootstrapNAS specific specific
        param_dict = self.manager.translate2param(x)
        sample = copy.deepcopy(param_dict)
        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'macs', 'accuracy_top1']:
            individual_results[metric] = 0

        # Predictor Mode
        if self.predictor_mode == True:
            if 'accuracy_top1' in self.optimization_metrics:
                individual_results['accuracy_top1'] = self.evaluator.estimate_accuracy_top1(
                    self.manager.onehot_generic(np_x).reshape(1, -1)
                )[0]
            if 'params' in self.optimization_metrics:
                individual_results['params'] = self.evaluator.estimate_parameters(
                    self.manager.onehot_generic(np_x).reshape(1, -1)
                )[0]
            if 'latency' in self.optimization_metrics:
                individual_results['latency'] = self.evaluator.estimate_latency(
                    self.manager.onehot_generic(np_x).reshape(1, -1)
                )[0]
            if 'macs' in self.optimization_metrics:
                individual_results['macs'] = self.evaluator.estimate_macs(
                    self.manager.onehot_generic(np_x).reshape(1, -1)
                )[0]

        # Validation Mode
        else:
            if 'accuracy_top1' in self.measurements:
                individual_results['accuracy_top1'] = self.evaluator.validate_top1(x)
            if 'macs' in self.measurements or 'params' in self.measurements:
                individual_results['macs'], individual_results['params'] = self.evaluator.validate_macs_params(x)
            if 'latency' in self.measurements:
                # TODO(macsz) Make sure that the input size is correct
                individual_results['latency'], _ = self.evaluator.measure_latency(x)

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
