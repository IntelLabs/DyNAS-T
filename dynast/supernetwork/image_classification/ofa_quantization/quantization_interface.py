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
import os
import shutil
import uuid
from datetime import datetime
from typing import Tuple

import numpy as np
import torch

from dynast.measure.latency import auto_steps
from dynast.predictors.dynamic_predictor import Predictor
from dynast.search.evaluation_interface import EvaluationInterface
from dynast.supernetwork.image_classification.ofa.ofa import model_zoo as ofa_model_zoo
from dynast.supernetwork.image_classification.ofa.ofa.imagenet_classification.data_providers.imagenet import (
    ImagenetDataProvider,
)
from dynast.supernetwork.image_classification.ofa.ofa.imagenet_classification.run_manager import (
    ImagenetRunConfig,
    RunManager,
)
from dynast.utils import log
from dynast.utils.datasets import ImageNet
from dynast.utils.nn import get_macs, get_parameters, measure_latency, validate_classification

from .depth_parser import DepthParser
from .inc_quantization import inc_qconfig_dict, inc_quantize


class Quantization:
    def __init__(self, calibration_dataloader=None, mp_calibration_samples=None):
        super(Quantization, self).__init__()
        self.calibration_dataloader = calibration_dataloader
        self.mp_calibration_samples = mp_calibration_samples

    def quantize(self, model, regex_module_names, masks, qparam_dict):
        indices = np.where(np.array(masks))
        for key in qparam_dict.keys():
            assert len(masks) == len(qparam_dict[key])
            qparam_dict[key] = np.array(qparam_dict[key])[indices].tolist()
            assert len(regex_module_names) == len(qparam_dict[key])

        qconfig_dict = inc_qconfig_dict(
            regex_module_names=regex_module_names,
            q_weights_bit=qparam_dict['q_weights_bit'],
            q_activations_bit=qparam_dict['q_activations_bit'],
            q_weights_mode=qparam_dict['q_weights_mode'],
            q_activations_mode=qparam_dict['q_activations_mode'],
            q_weights_granularity=qparam_dict['q_weights_granularity'],
            q_activations_granularity=qparam_dict['q_activations_granularity'],
        )

        model_qt = inc_quantize(
            model,
            qconfig_dict,
            data_loader=self.calibration_dataloader,
            mp_calibration_samples=self.mp_calibration_samples,
        )

        return model_qt


class QuantizedOFARunner:
    """The QuantizedOFARunner class manages the sub-network selection from the OFA super-network and
    the validation measurements of the sub-networks. ResNet50, MobileNetV3 w1.0, and MobileNetV3 w1.2
    are currently supported. Imagenet is required for these super-networks `imagenet-ilsvrc2012`.
    """

    def __init__(
        self,
        supernet: str,
        dataset_path: str,
        acc_predictor: Predictor = None,
        model_size_predictor: Predictor = None,
        latency_predictor: Predictor = None,
        params_predictor: Predictor = None,
        batch_size: int = 128,
        eval_batch_size: int = 128,
        mp_calibration_samples: int = 100,
        dataloader_workers: int = 4,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        verbose: bool = False,
    ):
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.model_size_predictor = model_size_predictor
        self.latency_predictor = latency_predictor
        self.params_predictor = params_predictor
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.test_fraction = test_fraction
        self.dataset_path = dataset_path
        self.mp_calibration_samples = mp_calibration_samples
        self.dataloader_workers = dataloader_workers
        self.verbose = verbose
        ImagenetDataProvider.DEFAULT_PATH = dataset_path
        if supernet == 'inc_quantization_ofa_resnet50':
            self.ofa_network = ofa_model_zoo.ofa_net('ofa_resnet50', pretrained=True)
        else:
            self.ofa_network = ofa_model_zoo.ofa_net(supernet, pretrained=True)

        self.run_config = ImagenetRunConfig(
            test_batch_size=eval_batch_size,
            n_worker=dataloader_workers,
        )
        self.depth_parser = DepthParser(supernet='ofa_resnet50', supernet_depth=[2] * 5, base_blocks=[2, 2, 4, 2])

        self._init_data()

        self.quantizer = Quantization(
            calibration_dataloader=self.calibration_dataloader, mp_calibration_samples=self.mp_calibration_samples
        )

    def _init_data(self):
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

    def estimate_accuracy_top1(self, subnet_cfg) -> float:
        top1 = self.acc_predictor.predict(subnet_cfg)
        return top1

    def estimate_model_size(self, subnet_cfg) -> int:
        model_size = self.model_size_predictor.predict(subnet_cfg)
        return model_size

    def estimate_latency(self, subnet_cfg) -> float:
        latency = self.latency_predictor.predict(subnet_cfg)
        return latency

    def estimate_parameters(self, subnet_cfg) -> int:
        parameters = self.params_predictor.predict(subnet_cfg)
        return parameters

    def quantize_and_calibrate(self, subnet, subnet_cfg):
        masks, regex_module_names = self.depth_parser.layer_parse(subnet_depth=subnet_cfg['d'], subnet_model=subnet)

        qparam_len = len(subnet_cfg['q_bits'])
        subnet_qt = self.quantizer.quantize(
            subnet,
            regex_module_names=regex_module_names,
            masks=masks,
            qparam_dict={
                'q_weights_bit': subnet_cfg['q_bits'],
                'q_activations_bit': subnet_cfg['q_bits'],
                'q_weights_mode': subnet_cfg['q_weights_mode'],
                'q_activations_mode': ['asymmetric'] * qparam_len,
                'q_weights_granularity': ['perchannel'] * qparam_len,
                'q_activations_granularity': ['pertensor'] * qparam_len,
            },
        )
        return subnet_qt

    def validate_quantized_top1(self, subnet_cfg, device=None) -> float:
        device = self.device if not device else device

        subnet = self.get_subnet(subnet_cfg)
        folder_name = '/tmp/ofa-tmp-{}'.format(uuid.uuid1().hex)  # TODO(macsz) root directory should be configurable
        run_manager = RunManager(
            '{}/eval_subnet'.format(folder_name),
            subnet,
            self.run_config,
            init=False,
            verbose=self.verbose,
        )
        run_manager.reset_running_statistics(net=subnet)

        subnet_qt = self.quantize_and_calibrate(subnet, subnet_cfg)

        # Test sampled subnet
        self.run_config.data_provider.assign_active_img_size(subnet_cfg['r'][0])

        loss, top1, top5 = validate_classification(
            model=subnet_qt,
            data_loader=self.dataloader,
            device=self.device,
        )

        return top1

    def validate_params(self, subnet_cfg: dict, device: str = None) -> Tuple[int, int]:
        """Measure Torch model's FLOPs/MACs as per FVCore calculation
        Args:
            subnet_cfg: sub-network Torch model
            device: Target device. If not provided will use self.device.
        Returns:
            `params`
        """
        device = self.device if not device else device

        model = self.get_subnet(subnet_cfg)

        # Get sum of model parameters
        params = get_parameters(
            model=model,
            device=device,
        )

        return params

    def measure_modelsize(self, subnet_cfg: dict, device: str = None) -> Tuple[int, int]:
        """Measure Torch model's FLOPs/MACs as per FVCore calculation
        Args:
            subnet_cfg: sub-network Torch model
            device: Target device. If not provided will use self.device.
        Returns:
            `params`
        """
        time_stamp = str(datetime.now())

        temp_name = f'temp_{time_stamp}'
        subnet = self.get_subnet(subnet_cfg)
        subnet_qt = self.quantize_and_calibrate(subnet, subnet_cfg)

        subnet_qt.save(temp_name)
        model_size = os.path.getsize(f'{temp_name}/best_model.pt') / 1e6

        shutil.rmtree(temp_name)

        return model_size

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_cfg: dict,
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
        input_size: tuple = (self.batch_size, 3, 224, 224)

        if not warmup_steps:
            warmup_steps = auto_steps(self.batch_size, is_warmup=True)
        if not measure_steps:
            measure_steps = auto_steps(self.batch_size)

        subnet = self.get_subnet(subnet_cfg)
        subnet_qt = self.quantize_and_calibrate(subnet, subnet_cfg)

        latency_mean, latency_std = measure_latency(
            model=subnet_qt,
            input_size=input_size,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            device=device,
        )
        return latency_mean, latency_std

    def get_subnet(self, subnet_cfg):
        if self.supernet == 'ofa_resnet50' or self.supernet == 'inc_quantization_ofa_resnet50':
            self.ofa_network.set_active_subnet(d=subnet_cfg['d'], e=subnet_cfg['e'], w=subnet_cfg['w'])
        else:
            self.ofa_network.set_active_subnet(ks=subnet_cfg['ks'], e=subnet_cfg['e'], d=subnet_cfg['d'])

        self.subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
        self.subnet.eval()
        return self.subnet


class EvaluationInterfaceQuantizedOFAResNet50(EvaluationInterface):
    def __init__(
        self,
        evaluator,
        manager,
        optimization_metrics: list = ['accuracy_top1', 'model_size'],
        measurements: list = ['accuracy_top1', 'model_size'],
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
        sample = {
            'wid': None,
            'd': param_dict['d'],
            'e': param_dict['e'],
            'w': param_dict['w'],
            'r': [224],
            'q_bits': param_dict['q_bits'],
            'q_weights_mode': param_dict['q_weights_mode'],
        }
        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'model_size', 'accuracy_top1']:
            individual_results[metric] = 0

        # Predictor Mode
        if self.predictor_mode == True:
            if 'params' in self.optimization_metrics:
                individual_results['params'] = self.evaluator.estimate_parameters(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]
            if 'latency' in self.optimization_metrics:
                individual_results['latency'] = self.evaluator.estimate_latency(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]
            if 'model_size' in self.optimization_metrics:
                individual_results['model_size'] = self.evaluator.estimate_model_size(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]
            if 'accuracy_top1' in self.optimization_metrics:
                individual_results['accuracy_top1'] = self.evaluator.estimate_accuracy_top1(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]

        # Validation Mode
        else:
            if 'model_size' in self.measurements:
                individual_results['model_size'] = self.evaluator.measure_modelsize(subnet_sample)

            if 'params' in self.measurements:
                individual_results['params'] = self.evaluator.validate_params(subnet_sample)
            if 'latency' in self.measurements:
                individual_results['latency'], _ = self.evaluator.measure_latency(
                    subnet_sample,
                    measure_steps=100,
                )  # For quantization latency stability it's better to measure over more steps

            if 'accuracy_top1' in self.measurements:
                individual_results['accuracy_top1'] = self.evaluator.validate_quantized_top1(subnet_sample)

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
                    individual_results['model_size'],
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
