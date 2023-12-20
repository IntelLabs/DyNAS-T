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
import random
import shutil
import string
import warnings
import datetime

import torch
from fvcore.nn import FlopCountAnalysis
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
from neural_compressor.quantization import fit

from dynast.search.evaluation_interface import EvaluationInterface
from dynast.supernetwork.multi_domain_networks.beit3_supernetwork import SuperBEiT3ForImageClassification
from dynast.supernetwork.multi_domain_networks.dataset import create_downstream_dataset
from dynast.supernetwork.multi_domain_networks.engine_for_elastic_finetuning import evaluate, get_handler
from dynast.supernetwork.multi_domain_networks.modeling_utils import _get_base_config
from dynast.supernetwork.multi_domain_networks.simplify_beit3_eval import get_args
from dynast.supernetwork.multi_domain_networks.utils import *
from dynast.utils import log
from dynast.utils.distributed import get_distributed_vars
from dynast.utils.nn import measure_latency

warnings.filterwarnings("ignore")


def get_regex_names(model):
    module_names = []
    regex_module_names = []
    for name, module in model.named_modules():
        if name.endswith('.layer'):  # and name!="bert.encoder.layer": #type(module) in (nn.modules.conv.Conv2d,) and
            module_names.append(name)
    for name in module_names:
        if 'A' in name.split('.'):
            regex_module_names.append(name)

    return regex_module_names


def validate_accuracy_top1(model, data_loader_test, task_handler, device):
    ext_test_stats, task_key = evaluate(
        data_loader_test, model, device, task_handler
    )  # ,search_space_choices,supernet_config)
    log.debug(
        f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%"
    )
    return ext_test_stats[task_key]


def compute_macs(model, sample_config, device):
    model.set_sample_config(sample_config)
    numels = []
    for module_name, module in model.named_modules():
        if hasattr(module, 'calc_sampled_param_num'):
            if module_name == 'classifier':
                continue
            if module_name.split('.')[1] == 'encoder':
                if int(module_name.split('.')[3]) > (sample_config['num_layers'] - 1):
                    continue

            numels.append(module.calc_sampled_param_num())
    params = sum(numels)

    input_image = torch.randn((1, 3, 224, 224), device=device)
    for module in model.modules():
        if hasattr(module, 'profile') and model != module:
            module.profile(True)
    flops = FlopCountAnalysis(model, input_image)
    macs = flops.total()

    for module in model.modules():
        if hasattr(module, 'profile') and model != module:
            module.profile(False)
    return macs, params


class Beit3ImageNetRunner:
    """The BertSST2Runner class manages the sub-network selection from the BERT super-network and
    the validation measurements of the sub-networks. Bert-Base network finetuned on SST-2 dataset is
    currently supported.
    """

    def __init__(
        self,
        supernet,
        dataset_path,
        acc_predictor=None,
        macs_predictor=None,
        latency_predictor=None,
        params_predictor=None,
        model_size_predictor=None,
        batch_size: int = 16,
        eval_batch_size: int = 16,
        checkpoint_path=None,
        device: str = 'cpu',
        mixed_precision: bool = False,
        mp_calibration_samples: int = 100,
    ):
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.latency_predictor = latency_predictor
        self.params_predictor = params_predictor
        self.model_size_predictor = model_size_predictor
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.mixed_precision = mixed_precision
        self.mp_calibration_samples = mp_calibration_samples

        if self.mixed_precision and 'cpu' not in self.device:
            log.warning(f'Mixed precision is not supported on "{self.device}". Setting to "cpu".')

        self.supernet_model, self.eval_dataloader, self.task_handler = self.create_model_dataset(self.checkpoint_path)

    def create_model_dataset(
        self,
        checkpoint_path: str,
        num_layers: int = 12,
    ):
        args_new, ds_init = get_args()
        args_new.data_path = self.dataset_path

        args_new.finetune = checkpoint_path
        args_new.model = 'beit3_base_patch16_224'
        args_new.task = 'imagenet'
        args_new.sentencepiece_model = 'dynast/supernetwork/multi_domain_networks/beit3.spm'  # TODO(macsz) Fix
        args_new.batch_size = self.batch_size
        args_new.eval_batch_size = self.eval_batch_size
        args_new.eval = True

        data_loader_test = create_downstream_dataset(args_new, is_eval=True)
        args_model = _get_base_config(drop_path_rate=args_new.drop_path)
        args_model.normalize_output = False
        args_model.encoder_layers = num_layers

        model = SuperBEiT3ForImageClassification(args_model, num_classes=1000)

        if args_new.finetune:
            load_model_and_may_interpolate(args_new.finetune, model, args_new.model_key, args_new.model_prefix)

        # torch.distributed.barrier()
        model_ema = None
        if args_new.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            if args_new.model_ema_force_cpu:
                self.device = 'cpu'
            model_ema = ModelEma(model, decay=args_new.model_ema_decay, device=self.device, resume='')
            log.debug("Using EMA with decay = %.8f" % args_new.model_ema_decay)
        # orch.distributed.init_process_group(backend='nccl',world_size=8,rank=0)
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        task_handler = get_handler(args_new)

        model.to(self.device)
        supernet_config = {"embed_size": 768, "num_layers": num_layers, "head_num": 12 * [12], "ffn_size": [3072] * 12}

        model.set_sample_config(supernet_config)
        return model, data_loader_test, task_handler

    def estimate_accuracy_top1(
        self,
        subnet_cfg: dict,
    ) -> float:
        top1 = self.acc_predictor.predict(subnet_cfg)
        return top1

    def estimate_macs(
        self,
        subnet_cfg: dict,
    ) -> int:
        macs = self.macs_predictor.predict(subnet_cfg)
        return macs

    def estimate_parameters(
        self,
        subnet_cfg: dict,
    ) -> int:
        return self.params_predictor.predict(subnet_cfg)

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

    def validate_accuracy_top1(
        self,
        subnet_cfg: dict,
        qbit_list,
    ) -> float:  # pragma: no cover
        model_fp32, _, _ = self.create_model_dataset(self.checkpoint_path)
        model_fp32.set_sample_config(subnet_cfg)

        if self.mixed_precision:
            regex_module_names = get_regex_names(self.supernet_model)
            quantized_model = self.quantize_subnet(model_fp32, qbit_list, regex_module_names)
            model_to_eval = quantized_model
        else:
            model_to_eval = model_fp32
            model_to_eval.to(self.device)

        accuracy_top1 = validate_accuracy_top1(model_to_eval, self.eval_dataloader, self.task_handler, self.device)
        del model_fp32
        if self.mixed_precision:
            del quantized_model

        return accuracy_top1

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
        macs, params = compute_macs(self.supernet_model, subnet_cfg, self.device)
        log.debug('Model\'s params: {}'.format(params))
        return macs, params

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_sample,
        qbit_list,
        warmup_steps: int = 10,
        measure_steps: int = 100,
    ):
        """Measure Torch model's latency.
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            mean latency; std latency
        """

        log.debug(
            f'Performing Latency measurements. Warmup = {warmup_steps}, Measure steps = {measure_steps}, Batch size = {self.batch_size}'
        )

        config_new = {
            'embed_size': 768,
            'num_layers': subnet_sample['num_layers'],
            'head_num': subnet_sample['head_num'][: subnet_sample['num_layers']],
            'ffn_size': subnet_sample["ffn_size"][: subnet_sample['num_layers']],
        }
        model_fp32, _, _ = self.create_model_dataset(self.checkpoint_path, num_layers=subnet_sample['num_layers'])

        model_fp32.set_sample_config(config_new)

        if self.mixed_precision:
            regex_module_names = get_regex_names(self.supernet_model)
            q_model = self.quantize_subnet_modelsize(model_fp32, qbit_list, regex_module_names)
            model_to_eval = q_model
        else:
            model_to_eval = model_fp32
            model_to_eval.to(self.device)

        lat_mean, lat_std = measure_latency(
            model_to_eval,
            input_size=(self.batch_size, 3, 224, 224),
            device=self.device,
        )
        del model_fp32
        if self.mixed_precision:
            del q_model

        log.debug('Model\'s latency: {} +/- {}'.format(lat_mean, lat_std))
        return lat_mean

    def quantize_subnet(
        self,
        model,
        qbit_list,
        regex_module_names,
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
            calibration_sampling_size=self.self.mp_calibration_samples,
            op_name_dict=q_config_dict,
        )

        q_model = fit(model, conf=conf, calib_dataloader=self.eval_dataloader)  # , eval_func=eval_func),
        del q_config_dict, conf, model
        return q_model

    def quantize_subnet_modelsize(
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
            approach="dynamic",
            tuning_criterion=tuning_criterion,
            calibration_sampling_size=self.self.mp_calibration_samples,
            op_name_dict=q_config_dict,
        )

        q_model = fit(model, conf=conf)
        del q_config_dict, conf, model
        return q_model

    def validate_modelsize(self, subnet_sample, qbit_list):
        LOCAL_RANK, WORLD_RANK, WORLD_SIZE, DIST_METHOD = get_distributed_vars()
        WORLD_RANK = WORLD_RANK if WORLD_RANK is not None else 0
        tmp_name = f"/tmp/dynast_{WORLD_RANK}_{''.join(random.choices(string.ascii_letters, k=6))}"
        # supernet_config =  {'subnet_hidden_sizes': 768,'num_layers': 12, 'num_attention_heads': [12]*12,
        #               'subnet_intermediate_sizes': [3072]*12}
        regex_module_names = get_regex_names(self.supernet_model)
        config_new = {
            'embed_size': 768,
            'num_layers': subnet_sample['num_layers'],
            'head_num': subnet_sample['head_num'][: subnet_sample['num_layers']],
            'ffn_size': subnet_sample["ffn_size"][: subnet_sample['num_layers']],
        }
        model_fp32, _, _ = self.create_model_dataset(self.checkpoint_path, num_layers=subnet_sample['num_layers'])

        model_fp32.set_sample_config(config_new)

        q_model = self.quantize_subnet_modelsize(model_fp32, qbit_list, regex_module_names)
        q_model.save(tmp_name)
        model_size = os.path.getsize(f'{tmp_name}/best_model.pt') / (1048576)
        log.debug('Size (MB):', model_size)

        shutil.rmtree(tmp_name)
        del q_model, model_fp32

        return model_size


class EvaluationInterfaceBeit3ImageNet(EvaluationInterface):
    def __init__(
        self,
        evaluator,
        manager,
        optimization_metrics: list = ['accuracy_top1', 'latency'],
        measurements: list = ['accuracy_top1', 'latency'],
        csv_path=None,
        predictor_mode: bool = False,
        mixed_precision: bool = False,
    ):
        super().__init__(
            evaluator, manager, optimization_metrics, measurements, csv_path, predictor_mode, mixed_precision
        )

    def eval_subnet(self, x):
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        if self.mixed_precision:
            qbit_list = param_dict['q_bits']
            self._set_workspace()
        else:
            qbit_list = None

        sample = {
            'embed_size': 768,
            'num_layers': param_dict['num_layers'][0],
            'head_num': param_dict['head_num'],
            'ffn_size': [3072] * 12,
        }

        subnet_sample = copy.deepcopy(sample)

        individual_results = dict()
        for metric in ['params', 'latency', 'macs', 'model_size', 'accuracy_top1']:
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
            if 'macs' in self.optimization_metrics:
                individual_results['macs'] = self.evaluator.estimate_macs(
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
            if 'macs' in self.measurements or 'params' in self.measurements:
                individual_results['macs'], individual_results['params'] = self.evaluator.validate_macs(subnet_sample)
            if 'latency' in self.measurements:
                individual_results['latency'] = self.evaluator.measure_latency(subnet_sample, qbit_list)
            if 'accuracy_top1' in self.measurements:
                individual_results['accuracy_top1'] = self.evaluator.validate_accuracy_top1(subnet_sample, qbit_list)
            if 'model_size' in self.measurements:
                individual_results['model_size'] = self.evaluator.validate_modelsize(subnet_sample, qbit_list)

        subnet_sample = param_dict
        sample = param_dict
        # Write result for csv_path
        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.datetime.now())
                result = [
                    subnet_sample,
                    date,
                    individual_results['params'],
                    individual_results['latency'],
                    individual_results['macs'],
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
