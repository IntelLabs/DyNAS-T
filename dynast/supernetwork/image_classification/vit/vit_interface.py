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
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torchprofile

from dynast.search.evaluation_interface import EvaluationInterface
from dynast.utils import log
from dynast.utils.datasets import ImageNet
from dynast.utils.nn import validate_classification

from .vit_supernetwork import SuperViT

warnings.filterwarnings("ignore")

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_CLASSES = 1000
DROPOUT = 0.1
ATTN_DROPOUT = 0.1

# ViT_B16
NUM_LAYERS_B_16 = 12
NUM_HEADS_B_16 = 12
HIDDEN_DIM_B_16 = 768
MLP_DIM_B_16 = 3072


def load_supernet(checkpoint_path):

    model = SuperViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_layers=NUM_LAYERS_B_16,
        num_heads=NUM_HEADS_B_16,
        hidden_dim=HIDDEN_DIM_B_16,
        mlp_dim=MLP_DIM_B_16,
        num_classes=NUM_CLASSES,
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location='cpu')['state_dict'],
        strict=True,
    )
    return model


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

# def compute_accuracy_imagenet(
#     config,
#     eval_dataloader,
#     model,
#     device: str = 'cpu',
# ):
#     """Measure ImageNet top@1, top@5 Accuracy scores of the ViT based model."""

#     model.eval()
#     model.to(device)
#     model.set_sample_config(config)

#     running_top1 = []
#     running_top5 = []
#     running_count = 0

#     for _, (images, target) in enumerate(eval_dataloader):

#         images = images.to(device)
#         target = target.to(device)

#         with torch.no_grad():
#             network_outputs = model(images)

#         acc1, acc5 = accuracy(network_outputs, target, topk=(1, 5))
#         running_top1.append(acc1[0].item() * images.size(0))
#         running_top5.append(acc5[0].item() * images.size(0))
#         running_count += images.size(0)

#     ave_top1 = np.sum(running_top1) / running_count
#     ave_top5 = np.sum(running_top5) / running_count

#     return ave_top1, ave_top5 # Return top5 if needed


def compute_val_acc(
    config,
    eval_dataloader,
    model,
    test_size,
    device: str = 'cpu',
):
    """Measure ImageNet top@1, top@5 Accuracy scores of the ViT based model."""

    model.eval()
    model.to(device)
    model.set_sample_config(config)
    return validate_classification(
        model=model, data_loader=eval_dataloader, epoch=0, test_size=test_size, device=device
    )


def compute_latency(
    config,
    model,
    eval_batch_size=4,
    device: str = 'cpu',
    warmup_steps: int = 10,
    measure_steps: int = 100,
):
    """Measure latency of the ViT-based model."""

    model.eval()
    model.to(device)
    model.set_sample_config(config)

    images = torch.zeros([eval_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=torch.float, device=device)

    latencies = []

    with torch.no_grad():
        for _ in range(warmup_steps):
            model(images)

        for _ in range(measure_steps):
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            start = time.time()
            model(images)
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            end = time.time()
            latencies.append((end - start) * 1e3)

    # Drop the first and last 5% latency numbers
    truncated_latency = np.array(latencies)[int(measure_steps * 0.05) : int(measure_steps * 0.95)]

    latency_mean = np.round(np.mean(truncated_latency), 3)
    latency_std = np.round(np.std(truncated_latency), 3)

    return latency_mean, latency_std


# TODO: Make this correct for ViT: Fix param computation
def compute_macs(config, model, base_config, device: str = 'cpu'):
    """Calculate MACs for ViT-based models."""

    model.eval()
    model.to(device)

    model.set_sample_config(config)

    # Compute MACS
    for module in model.modules():
        if hasattr(module, 'profile') and model != module:
            module.profile(True)

    images = torch.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=torch.float, device=device)

    macs = torchprofile.profile_macs(model, args=(images))

    for module in model.modules():
        if hasattr(module, 'profile') and model != module:
            module.profile(False)

    # Compute Params
    # base_hidden_size = base_config.hidden_size
    # embedding_params = (
    #     base_config.vocab_size * base_hidden_size
    #     + base_config.max_position_embeddings * base_hidden_size
    #     + base_config.type_vocab_size * base_hidden_size
    # )
    # numels = []

    # for module_name, module in model.named_modules():
    #     if hasattr(module, 'calc_sampled_param_num'):
    #         if module_name == 'classifier':
    #             continue
    #         if module_name.split('.')[1] == 'encoder':
    #             if int(module_name.split('.')[3]) > (config['num_layers'] - 1):
    #                 continue

    #         numels.append(module.calc_sampled_param_num())

    # params = sum(numels) + embedding_params
    params = 0
    return macs, params


class ViTRunner:
    """The ViTRunner class manages the sub-network selection from the BERT super-network and
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
        batch_size: int = 16,
        checkpoint_path=None,
        total_batches=None,
        device: str = 'cpu',
    ):

        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.latency_predictor = latency_predictor
        self.params_predictor = params_predictor
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.test_size = total_batches
        self.eval_dataloader = ImageNet.validation_dataloader(batch_size=self.batch_size)
        # TODO: Figure out if a similar base config can be created for ViT
        # self.supernet_model, self.base_config = load_supernet(self.checkpoint_path)
        self.supernet_model = load_supernet(self.checkpoint_path)

    def estimate_accuracy_imagenet(
        self,
        subnet_cfg: dict,
    ) -> float:
        top1 = self.acc_predictor.predict(subnet_cfg)
        return top1

    def estimate_macs(
        self,
        subnet_cfg: dict,
    ) -> int:
        # TODO: Fix mac computation
        macs = self.macs_predictor.predict(subnet_cfg)
        return macs

    def estimate_latency(
        self,
        subnet_cfg: dict,
    ) -> float:
        latency = self.latency_predictor.predict(subnet_cfg)
        return latency

    def validate_accuracy_imagenet(
        self,
        subnet_cfg: dict,
    ) -> float:  # pragma: no cover

        _, top1_accuracy, _ = compute_val_acc(
            config=subnet_cfg,
            eval_dataloader=self.eval_dataloader,
            model=self.supernet_model,
            test_size=self.test_size,
            device=self.device,
        )
        return top1_accuracy

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
        # TODO: Fix Macs computation
        self.base_config = None
        macs, params = compute_macs(subnet_cfg, self.supernet_model, self.base_config)
        logging.info('Model\'s macs: {}'.format(macs))
        return macs, params

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_cfg: dict,
        eval_batch_size=4,
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

        lat_mean, lat_std = compute_latency(subnet_cfg, self.supernet_model, eval_batch_size, device=self.device)
        logging.info('Model\'s latency: {} +/- {}'.format(lat_mean, lat_std))

        return lat_mean, lat_std


class EvaluationInterfaceViT(EvaluationInterface):
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
            if 'accuracy_top1' in self.optimization_metrics:
                individual_results['accuracy_top1'] = self.evaluator.estimate_accuracy_imagenet(
                    self.manager.onehot_custom(param_dict).reshape(1, -1)
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
                    individual_results['latency'],
                    individual_results['macs'],
                    individual_results['params'],
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
