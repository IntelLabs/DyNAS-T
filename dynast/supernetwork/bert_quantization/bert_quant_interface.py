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
import logging
import os
import shutil
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torchprofile
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
from neural_compressor.quantization import fit
from transformers import BertConfig

from dynast.search.evaluation_interface import EvaluationInterface
from dynast.utils import log

from .bert_subnetwork import BertSubnetForSequenceClassification
from .bert_supernetwork import BertSupernetForSequenceClassification
from .sst2_dataloader import prepare_calib_loader, prepare_data_loader

warnings.filterwarnings("ignore")


def get_weights_copy(model):
    weights_path = 'weights_temp.pt'
    torch.save(model.state_dict(), weights_path)
    return torch.load(weights_path)


def load_supernet(checkpoint_path):
    bert_config = BertConfig()
    model = BertSupernetForSequenceClassification(bert_config, num_labels=2)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location='cpu')["model"],
        strict=True,
    )
    return model, bert_config


def load_subnet(checkpoint_path, num_layers):
    bert_config = BertConfig()
    model = BertSubnetForSequenceClassification(bert_config, 2, num_layers)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location='cpu')["model"],
        strict=False,
    )
    return model, bert_config


def get_regex_names(model):
    regex_module_names = []
    for name, module in model.named_modules():
        # print(name)
        if name.endswith('.layer') and name != "bert.encoder.layer":  # type(module) in (nn.modules.conv.Conv2d,) and
            regex_module_names.append(name)
    return regex_module_names


def compute_accuracy_sst2(
    eval_dataloader,
    model,
    device: str = 'cpu',
):
    """Measure SST-2 Accuracy score of the BERT based model."""

    model.eval()
    # model.to(device)
    # model.set_sample_config(config)

    preds = None
    out_label_ids = None

    for i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            network_outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            logits = network_outputs[1]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids,
                label_ids.detach().cpu().numpy(),
                axis=0,
            )
    preds = np.argmax(preds, axis=1)
    accuracy_sst2 = (preds == out_label_ids).mean()

    return accuracy_sst2


def compute_latency(
    model,
    eval_batch_size=4,
    device: str = 'cpu',
    warmup_steps: int = 50,
    measure_steps: int = 500,
):
    """Measure latency of the BERT-based model."""

    input_ids = torch.zeros([eval_batch_size, 128], dtype=torch.long, device=device)
    segment_ids = torch.zeros([eval_batch_size, 128], dtype=torch.long, device=device)
    input_mask = torch.zeros([eval_batch_size, 128], dtype=torch.long, device=device)

    latencies = []

    with torch.no_grad():
        for i in range(warmup_steps):
            model(input_ids, input_mask, segment_ids)

        for i in range(measure_steps):
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            start = time.time()
            model(input_ids, input_mask, segment_ids)
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            end = time.time()
            latencies.append((end - start) * 1e3)

    # Drop the first and last 5% latency numbers
    truncated_latency = np.array(latencies)[int(measure_steps * 0.05) : int(measure_steps * 0.95)]

    latency_mean = np.round(np.mean(truncated_latency), 3)
    latency_std = np.round(np.std(truncated_latency), 3)

    return latency_mean, latency_std


class BertSST2QuantizedRunner:
    """The BertSST2Runner class manages the sub-network selection from the BERT super-network and
    the validation measurements of the sub-networks. Bert-Base network finetuned on SST-2 dataset is
    currently supported.
    """

    def __init__(
        self,
        supernet,
        dataset_path,
        acc_predictor=None,
        model_size_predictor=None,
        latency_predictor=None,
        params_predictor=None,
        batch_size: int = 16,
        checkpoint_path=None,
        device: str = 'cpu',
    ):
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.model_size_predictor = model_size_predictor
        self.latency_predictor = latency_predictor
        self.params_predictor = params_predictor
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.supernet_model, self.base_config = load_supernet(self.checkpoint_path)
        supernet_config = {
            'subnet_hidden_sizes': 768,
            'num_layers': 12,
            'num_attention_heads': [12] * 12,
            'subnet_intermediate_sizes': [3072] * 12,
        }

        model_new = self.supernet_model
        self.eval_dataloader = prepare_data_loader(self.dataset_path)
        self.calib_dataloader = prepare_calib_loader(self.dataset_path, model_new, eval_batch_size=16)

        self.supernet_model.set_sample_config(supernet_config)

    def estimate_accuracy_sst2(
        self,
        subnet_cfg: dict,
    ) -> float:
        top1 = self.acc_predictor.predict(subnet_cfg)
        return top1

    def estimate_model_size(
        self,
        subnet_cfg: dict,
    ) -> int:
        model_size = self.model_size_predictor.predict(subnet_cfg)
        return model_size

    def estimate_latency(
        self,
        subnet_cfg: dict,
    ) -> float:
        latency = self.latency_predictor.predict(subnet_cfg)
        return latency

    def validate_accuracy_sst2(
        self,
        subnet_sample,
        qbit_list,
    ) -> float:  # pragma: no cover
        regex_module_names = get_regex_names(self.supernet_model)
        supernet_config = {
            'subnet_hidden_sizes': 768,
            'num_layers': 12,
            'num_attention_heads': [12] * 12,
            'subnet_intermediate_sizes': [3072] * 12,
        }
        model_fp32, _ = load_supernet(self.checkpoint_path)
        model_fp32.set_sample_config(supernet_config)
        model_fp32.set_sample_config(subnet_sample)
        quantized_model = self.quantize_subnet(model_fp32, qbit_list, regex_module_names)
        accuracy_sst2 = compute_accuracy_sst2(self.eval_dataloader, quantized_model, device=self.device)
        del quantized_model, model_fp32
        return accuracy_sst2

    def validate_macs(
        self,
        subnet_cfg: dict,
    ) -> float:
        """Measure Torch model's FLOPs/MACs as per FVCore calculation
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            `macs`
        """
        macs, params = compute_macs(subnet_cfg, self.supernet_model, self.base_config)
        logging.info('Model\'s macs: {}'.format(macs))
        return macs, params

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_sample: dict,
        qbit_list,
        eval_batch_size=128,
        warmup_steps: int = 10,
        measure_steps: int = 100,
    ):
        """Measure Torch model's latency.
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            mean latency; std latency
        """

        logging.info(
            f'Performing Latency measurements. Warmup = {warmup_steps},\
             Measure steps = {measure_steps}'
        )

        regex_module_names = get_regex_names(self.supernet_model)
        config_new = {
            'subnet_hidden_sizes': 768,
            'num_layers': subnet_sample['num_layers'],
            'num_attention_heads': subnet_sample['num_attention_heads'][: subnet_sample['num_layers']],
            'subnet_intermediate_sizes': subnet_sample["subnet_intermediate_sizes"][: subnet_sample['num_layers']],
        }
        model_fp32, _ = load_subnet(self.checkpoint_path, subnet_sample['num_layers'])
        model_fp32.set_sample_config(config_new)

        q_model = self.quantize_subnet(model_fp32, qbit_list, regex_module_names)
        lat_mean, lat_std = compute_latency(q_model, eval_batch_size, device=self.device)
        logging.info('Model\'s latency: {} +/- {}'.format(lat_mean, lat_std))

        return lat_mean, lat_std

    def validate_modelsize(self, subnet_sample, qbit_list):
        temp_name = "temp_23"
        # supernet_config =  {'subnet_hidden_sizes': 768,'num_layers': 12, 'num_attention_heads': [12]*12,
        #               'subnet_intermediate_sizes': [3072]*12}
        regex_module_names = get_regex_names(self.supernet_model)
        config_new = {
            'subnet_hidden_sizes': 768,
            'num_layers': subnet_sample['num_layers'],
            'num_attention_heads': subnet_sample['num_attention_heads'][: subnet_sample['num_layers']],
            'subnet_intermediate_sizes': subnet_sample["subnet_intermediate_sizes"][: subnet_sample['num_layers']],
        }
        model_fp32, _ = load_subnet(self.checkpoint_path, subnet_sample['num_layers'])
        # model_fp32.set_sample_config(supernet_config)
        model_fp32.set_sample_config(config_new)

        q_model = self.quantize_subnet(model_fp32, qbit_list, regex_module_names)
        q_model.save(temp_name)
        model_size = os.path.getsize(f'{temp_name}/best_model.pt') / 1e6
        print('Size (MB):', model_size)

        shutil.rmtree(temp_name)
        del q_model, model_fp32
        return model_size

    def quantize_subnet(
        self,
        model,
        qbit_list,
        regex_module_names,
        # calib_dataloader,
    ):
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
            calibration_sampling_size=16,
            op_name_dict=q_config_dict,
        )

        q_model = fit(model, conf=conf, calib_dataloader=self.calib_dataloader)  # , eval_func=eval_func),
        del q_config_dict, conf, model
        return q_model


class EvaluationInterfaceBertSST2Quantized(EvaluationInterface):
    def __init__(
        self,
        evaluator,
        manager,
        optimization_metrics: list = ['accuracy_sst2', 'latency'],
        measurements: list = ['accuracy_sst2', 'latency'],
        csv_path=None,
        predictor_mode: bool = False,
    ):
        super().__init__(evaluator, manager, optimization_metrics, measurements, csv_path, predictor_mode)

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)
        qbit_list = param_dict['q_bits']
        sample = {
            'subnet_hidden_sizes': 768,
            'num_layers': param_dict['num_layers'][0],
            'num_attention_heads': param_dict['num_attention_heads'],
            'subnet_intermediate_sizes': param_dict["intermediate_size"],
        }

        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'model_size', 'accuracy_sst2']:
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
                # import ipdb;ipdb.set_trace()
                individual_results['model_size'] = self.evaluator.estimate_model_size(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]
            if 'accuracy_sst2' in self.optimization_metrics:
                # import ipdb;ipdb.set_trace()
                individual_results['accuracy_sst2'] = self.evaluator.estimate_accuracy_sst2(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
                )[0]

        # Validation Mode
        else:
            if 'params' in self.measurements:
                individual_results['params'] = 0  # self.evaluator.validate_macs(subnet_sample)

            if 'accuracy_sst2' in self.measurements:
                individual_results['accuracy_sst2'] = self.evaluator.validate_accuracy_sst2(subnet_sample, qbit_list)

            if 'model_size' in self.measurements:
                individual_results['model_size'] = self.evaluator.validate_modelsize(subnet_sample, qbit_list)
            if 'latency' in self.measurements:
                individual_results['latency'], _ = self.evaluator.measure_latency(
                    subnet_sample, qbit_list, eval_batch_size=16
                )

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
                    individual_results['accuracy_sst2'],
                ]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        individual_results['accuracy_sst2'] = -individual_results['accuracy_sst2']
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
