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
import uuid
from datetime import datetime
from typing import Tuple

import ofa
import torch
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.tutorial.flops_table import rm_bn_from_net

from dynast.measure.latency import auto_steps
from dynast.predictors.dynamic_predictor import Predictor
from dynast.search.evaluation_interface import EvaluationInterface
from dynast.utils import log
from dynast.utils.nn import get_macs, get_parameters, measure_latency


class OFARunner:
    """The OFARunner class manages the sub-network selection from the OFA super-network and
    the validation measurements of the sub-networks. ResNet50, MobileNetV3 w1.0, and MobileNetV3 w1.2
    are currently supported. Imagenet is required for these super-networks `imagenet-ilsvrc2012`.
    """

    def __init__(
        self,
        supernet: str,
        dataset_path: str,
        acc_predictor: Predictor = None,
        macs_predictor: Predictor = None,
        latency_predictor: Predictor = None,
        params_predictor: Predictor = None,
        batch_size: int = 1,
        dataloader_workers: int = 4,
        device: str = 'cpu',
        valid_size: int = None,
    ):
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.latency_predictor = latency_predictor
        self.params_predictor = params_predictor
        self.batch_size = batch_size
        self.device = device
        self.test_size = None
        ImagenetDataProvider.DEFAULT_PATH = dataset_path
        self.ofa_network = ofa.model_zoo.ofa_net(supernet, pretrained=True)
        self.run_config = ImagenetRunConfig(
            test_batch_size=batch_size,
            n_worker=dataloader_workers,
            valid_size=valid_size,
        )

    def estimate_accuracy_top1(self, subnet_cfg) -> float:
        top1 = self.acc_predictor.predict(subnet_cfg)
        return top1

    def estimate_macs(self, subnet_cfg) -> int:
        macs = self.macs_predictor.predict(subnet_cfg)
        return macs

    def estimate_latency(self, subnet_cfg) -> float:
        latency = self.latency_predictor.predict(subnet_cfg)
        return latency

    def estimate_parameters(self, subnet_cfg) -> int:
        parameters = self.params_predictor.predict(subnet_cfg)
        return parameters

    def validate_top1(self, subnet_cfg, device=None) -> float:
        device = self.device if not device else device

        subnet = self.get_subnet(subnet_cfg)
        folder_name = '/tmp/ofa-tmp-{}'.format(uuid.uuid1().hex)  # TODO(macsz) root directory should be configurable
        run_manager = RunManager('{}/eval_subnet'.format(folder_name), subnet, self.run_config, init=False)
        run_manager.reset_running_statistics(net=subnet)

        # Test sampled subnet
        self.run_config.data_provider.assign_active_img_size(subnet_cfg['r'][0])
        loss, acc = run_manager.validate(net=subnet, no_logs=True)
        top1 = acc[0]
        return top1

    def validate_macs_params(self, subnet_cfg: dict, device: str = None) -> Tuple[int, int]:
        """Measure Torch model's FLOPs/MACs as per FVCore calculation
        Args:
            subnet_cfg: sub-network Torch model
            device: Target device. If not provided will use self.device.
        Returns:
            `macs`
        """
        device = self.device if not device else device

        model = self.get_subnet(subnet_cfg)

        # MACs=FLOPs in FVCore
        macs = get_macs(
            model=model,
            input_size=(1, 3, 224, 224),  # batch size does not matter for MACs (scales linearly).
            device=device,
        )

        # Get sum of model parameters
        params = get_parameters(
            model=model,
            device=device,
        )

        return macs, params

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_cfg: dict,
        input_size: tuple = (1, 3, 224, 224),
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
        device = self.device if not device else device

        if not warmup_steps:
            warmup_steps = auto_steps(input_size[0], is_warmup=True)
        if not measure_steps:
            measure_steps = auto_steps(input_size[0])

        model = self.get_subnet(subnet_cfg)

        latency_mean, latency_std = measure_latency(
            model=model,
            input_size=input_size,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            device=device,
        )
        return latency_mean, latency_std

    def get_subnet(self, subnet_cfg):
        if self.supernet == 'ofa_resnet50':
            self.ofa_network.set_active_subnet(ks=subnet_cfg['d'], e=subnet_cfg['e'], d=subnet_cfg['w'])
        else:
            self.ofa_network.set_active_subnet(ks=subnet_cfg['ks'], e=subnet_cfg['e'], d=subnet_cfg['d'])

        self.subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
        self.subnet.eval()
        return self.subnet


class EvaluationInterfaceOFAResNet50(EvaluationInterface):
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

        # PyMoo vector to Super-Network Parameter Mapping, OFA specific
        param_dict = self.manager.translate2param(x)
        sample = {'d': param_dict['d'], 'e': param_dict['e'], 'w': param_dict['w'], 'r': [224]}
        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'macs', 'accuracy_top1']:
            individual_results[metric] = 0

        # Predictor Mode
        if self.predictor_mode == True:
            if 'params' in self.optimization_metrics:
                individual_results['params'] = self.evaluator.estimate_parameters(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]
            if 'latency' in self.optimization_metrics:
                individual_results['latency'] = self.evaluator.estimate_latency(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]
            if 'macs' in self.optimization_metrics:
                individual_results['macs'] = self.evaluator.estimate_macs(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]
            if 'accuracy_top1' in self.optimization_metrics:
                individual_results['accuracy_top1'] = self.evaluator.estimate_accuracy_top1(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]

        # Validation Mode
        else:
            if 'macs' in self.measurements or 'params' in self.measurements:
                individual_results['macs'], individual_results['params'] = self.evaluator.validate_macs_params(
                    subnet_sample
                )
            if 'latency' in self.measurements:
                individual_results['latency'], _ = self.evaluator.measure_latency(
                    subnet_sample
                )  # TODO(change batch size!!)
            if 'accuracy_top1' in self.measurements:
                individual_results['accuracy_top1'] = self.evaluator.validate_top1(subnet_sample)

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


class EvaluationInterfaceOFAMobileNetV3(EvaluationInterface):
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

        # PyMoo vector to Super-Network Parameter Mapping, OFA specific
        param_dict = self.manager.translate2param(x)
        sample = {'wid': None, 'ks': param_dict['ks'], 'e': param_dict['e'], 'd': param_dict['d'], 'r': [224]}
        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'macs', 'accuracy_top1']:
            individual_results[metric] = 0

        # Predictor Mode
        if self.predictor_mode == True:
            if 'params' in self.optimization_metrics:
                individual_results['params'] = self.evaluator.estimate_parameters(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]
            if 'latency' in self.optimization_metrics:
                individual_results['latency'] = self.evaluator.estimate_latency(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]
            if 'macs' in self.optimization_metrics:
                individual_results['macs'] = self.evaluator.estimate_macs(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]
            if 'accuracy_top1' in self.optimization_metrics:
                individual_results['accuracy_top1'] = self.evaluator.estimate_accuracy_top1(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]

        # Validation Mode
        else:
            if 'macs' in self.measurements or 'params' in self.measurements:
                individual_results['macs'], individual_results['params'] = self.evaluator.validate_macs_params(
                    subnet_sample
                )
            if 'latency' in self.measurements:
                individual_results['latency'], _ = self.evaluator.measure_latency(subnet_sample)
            if 'accuracy_top1' in self.measurements:
                individual_results['accuracy_top1'] = self.evaluator.validate_top1(subnet_sample)

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
