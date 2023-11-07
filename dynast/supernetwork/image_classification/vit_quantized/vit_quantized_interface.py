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
from datetime import datetime
from typing import Optional

import torch
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
from neural_compressor.model.torch_model import PyTorchFXModel
from neural_compressor.quantization import fit

from dynast.predictors.dynamic_predictor import Predictor
from dynast.search.evaluation_interface import EvaluationInterface
from dynast.supernetwork.image_classification.ofa_quantization.quantization_interface import Quantization
from dynast.supernetwork.image_classification.vit.vit_interface import ViTRunner, load_supernet
from dynast.supernetwork.image_classification.vit_quantized.vit_quantized_encoding import ViTQuantizedEncoding
from dynast.utils import log, measure_time
from dynast.utils.datasets import ImageNet


def get_regex_names(model):
    regex_module_names = []
    for name, module in model.named_modules():
        # if name.endswith('.query') \
        #     or name.endswith('.key') \
        #     or name.endswith('.value') \
        #     or name.endswith('.dense') \
        #     or 'mlp.linear' in name \
        #     or '.ln_' in name:
        # if (
        #     'mlp.linear_' in name \
        #     or name.endswith('query') \
        #     or name.endswith('key') \
        #     or name.endswith('value') \
        #     or name.endswith('dense') \
        #     or 'encoder.ln' in name \
        #     or 'heads.head' in name
        # ):
        if name.endswith('layer') or 'encoder.ln' in name or 'heads.head' in name:
            regex_module_names.append(name)
            # print(">>>", name)
        else:
            pass
            # print(name)
    # print(f'{len(regex_module_names)=}')
    # print(f'{regex_module_names=}')
    # exit()

    log.debug(f'Matched {len(regex_module_names)} layers for Quantization: {regex_module_names}')
    return regex_module_names


class ViTQuantizedRunner(ViTRunner):
    def __init__(
        self,
        supernet,
        dataset_path,
        acc_predictor: Optional[Predictor] = None,
        model_size_predictor: Optional[Predictor] = None,
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
        self.model_size_predictor = model_size_predictor
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

        self.reload_supernet()

        self._init_data()

        self.quantizer = Quantization(
            calibration_dataloader=self.calibration_dataloader, mp_calibration_samples=self.mp_calibration_samples
        )

    def reload_supernet(self):
        log.debug(f'Reloading supernetwork from {self.checkpoint_path}')
        self.supernet_model, self.max_layers = load_supernet(self.checkpoint_path)

    def activate_subnet(self, subnet_config: dict) -> None:
        log.debug(f'Activating subnet with config: {subnet_config}')
        self.supernet_model.set_sample_config(subnet_config)

    @measure_time
    def quantize_subnet(self, subnet_config: dict, qbit_list: list):
        log.debug('Applying quantization policy on subnet.')
        self.reload_supernet()
        self.activate_subnet(subnet_config)
        # qbit_list = [8]*74

        regex_module_names = get_regex_names(self.supernet_model)

        default_config = {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}
        q_config_dict = {}
        count = 0
        # model_fp32 = copy.deepcopy(model)
        for mod_name in regex_module_names:
            q_config_dict[mod_name] = copy.deepcopy(default_config)
            # import ipdb;db.set_trace()
            if qbit_list[count] == 32:
                dtype = ['fp32']
            else:
                dtype = ['int8']

            q_config_dict[mod_name]['weight']['dtype'] = dtype
            q_config_dict[mod_name]['activation']['dtype'] = dtype
            count = count + 1
        tuning_criterion = TuningCriterion(max_trials=1)

        conf = PostTrainingQuantConfig(
            approach="static",
            tuning_criterion=tuning_criterion,
            calibration_sampling_size=16,  # TODO(macsz) `self.mp_calibration_samples`?
            op_name_dict=q_config_dict,
        )
        log.debug(f'Applying quantization policy: {conf}')
        q_model = fit(self.supernet_model, conf=conf, calib_dataloader=self.calibration_dataloader)
        log.debug('Quantization finished.')
        del (
            q_config_dict,
            conf,
            self.supernet_model,
        )
        return q_model

    def _init_data(self) -> None:
        ImageNet.PATH = self.dataset_path
        if self.dataset_path:
            self.eval_dataloader = ImageNet.validation_dataloader(
                batch_size=self.eval_batch_size,
                num_workers=self.dataloader_workers,
                fraction=self.test_fraction,
            )
            # TODO(macsz) Consider adding `train_batch_size`
            self.calibration_dataloader = ImageNet.train_dataloader(batch_size=self.eval_batch_size)

        else:
            self.eval_dataloader = None
            log.warning('No dataset path provided. Cannot validate sub-networks.')

    def estimate_model_size(self, subnet_cfg) -> int:
        model_size = self.model_size_predictor.predict(subnet_cfg) if self.model_size_predictor else -1
        return model_size

    def measure_model_size(
        self,
        model: PyTorchFXModel,
    ) -> float:
        tmp_name = 'temp.pt'  # TODO(macsz) Should be random.
        model.save(tmp_name)
        model_size = os.path.getsize(f'{tmp_name}/best_model.pt') / 1e6
        print('Size (MB):', model_size)

        shutil.rmtree(tmp_name)

        return model_size

    def quantize_and_calibrate(self, subnet, subnet_cfg):
        # TODO(macsz) Implement
        return None

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_cfg: dict,
    ):
        # TODO(macsz) Implement
        return -1

    def validate_accuracy_imagenet(
        self,
        subnet_cfg: dict,
    ) -> float:  # pragma: no cover
        # TODO(macsz) Implement
        return -1


class EvaluationInterfaceViTQuantized(EvaluationInterface):
    def __init__(
        self,
        evaluator: ViTQuantizedRunner,
        manager: ViTQuantizedEncoding,
        optimization_metrics: list = ['accuracy_top1', 'latency'],
        measurements: list = ['accuracy_top1', 'latency'],
        csv_path=None,
        predictor_mode: bool = False,
    ):
        super().__init__(evaluator, manager, optimization_metrics, measurements, csv_path, predictor_mode)

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        qbit_list = param_dict['q_bits']

        sample = {
            'vit_hidden_sizes': 768,
            'num_layers': param_dict['num_layers'][0],
            'num_attention_heads': param_dict['num_attention_heads'],
            'vit_intermediate_sizes': param_dict["vit_intermediate_sizes"],
        }

        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'model_size', 'accuracy_top1']:
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
            if 'model_size' in self.optimization_metrics:
                individual_results['model_size'] = self.evaluator.estimate_macs(
                    self.manager.onehot_custom(param_dict, max_layers=self.evaluator.max_layers).reshape(1, -1)
                )[0]
            if 'accuracy_top1' in self.optimization_metrics:
                individual_results['accuracy_top1'] = self.evaluator.estimate_accuracy_imagenet(
                    self.manager.onehot_custom(param_dict, max_layers=self.evaluator.max_layers).reshape(1, -1)
                )[0]

        # Validation Mode
        else:
            # TODO(macsz) Quantize `model` constructed with `subne_sample` w/ `qbit_list`
            #  subnet_config: dict, qbit_list: list, regex_module_names: list
            q_model = self.evaluator.quantize_subnet(
                subnet_config=subnet_sample,
                qbit_list=qbit_list,
            )

            if 'model_size' in self.measurements:
                individual_results['model_size'] = self.evaluator.measure_model_size(q_model)
            if 'latency' in self.measurements:
                individual_results['latency'] = self.evaluator.measure_latency(q_model)
            if 'accuracy_top1' in self.measurements:
                individual_results['accuracy_top1'] = self.evaluator.validate_accuracy_imagenet(q_model)
            if 'params' in self.measurements:
                _, individual_results['params'] = self.evaluator.validate_macs(q_model)

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
