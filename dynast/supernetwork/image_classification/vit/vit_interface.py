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
    max_layers = NUM_LAYERS_B_16

    model.load_state_dict(
        torch.load(checkpoint_path, map_location='cpu')['state_dict'],
        strict=True,
    )
    return model, max_layers


def compute_val_acc(
    config,
    eval_dataloader,
    model,
    device: str = 'cpu',
):
    """Measure ImageNet top@1, top@5 Accuracy scores of the ViT based model."""

    model.eval()
    model.to(device)
    model.set_sample_config(config)
    return validate_classification(
        model=model,
        data_loader=eval_dataloader,
        device=device,
    )


def compute_latency(
    config,
    model,
    batch_size=128,
    device: str = 'cpu',
    warmup_steps: int = 10,
    measure_steps: int = 100,
):
    """Measure latency of the ViT-based model."""

    # TODO(macsz) Use built-in methods

    model.eval()
    model.to(device)
    model.set_sample_config(config)

    input_size = (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    images = torch.zeros(input_size, dtype=torch.float, device=device)

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


def compute_macs(config, model, device: str = 'cpu'):
    """Calculate MACs for ViT-based models."""

    # TODO(macsz) Use built-in methods

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

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
        eval_batch_size: int = 128,
        checkpoint_path=None,
        device: str = 'cpu',
        test_fraction: float = 1.0,
        warmup_steps: int = 10,
        measure_steps: int = 100,
    ):
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

        self.supernet_model, self.max_layers = load_supernet(self.checkpoint_path)

        self._init_data()

    def _init_data(self):
        if self.dataset_path:
            ImageNet.PATH = self.dataset_path
            self.eval_dataloader = ImageNet.validation_dataloader(
                batch_size=self.eval_batch_size, fraction=self.test_fraction
            )
        else:
            self.dataloader = None
            log.warning('No dataset path provided. Cannot validate sub-networks.')

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
        macs = self.macs_predictor.predict(subnet_cfg)
        return macs

    def estimate_parameters(
        self,
        subnet_cfg: dict,
    ) -> int:
        parameters = self.params_predictor.predict(subnet_cfg)
        return parameters

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
        macs, params = compute_macs(subnet_cfg, self.supernet_model)
        logging.info('Model\'s macs: {}'.format(macs))
        return macs, params

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_cfg: dict,
    ):
        """Measure Torch model's latency.
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            mean latency; std latency
        """

        logging.info(
            f'Performing Latency measurements. Warmup = {self.warmup_steps},\
             Measure steps = {self.measure_steps}'
        )

        lat_mean, lat_std = compute_latency(
            config=subnet_cfg,
            model=self.supernet_model,
            batch_size=self.batch_size,
            device=self.device,
            warmup_steps=self.warmup_steps,
            measure_steps=self.measure_steps,
        )
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
