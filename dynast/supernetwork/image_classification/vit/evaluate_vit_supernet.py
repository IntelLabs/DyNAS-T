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
import yaml

from dynast.search.evaluation_interface import EvaluationInterface

# from dynast.supernetwork.image_classification.vit.vit_supernetwork import SuperViT
from dynast.supernetwork.image_classification.vit.vit_interface import ViTRunner
from dynast.utils import log
from dynast.utils.datasets import ImageNet
from dynast.utils.nn import validate_classification

# warnings.filterwarnings("ignore")


def config_parser(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    batch_size = 128
    checkpoint_path = "/localdisk/sairamsu/vit_21k_search/run_2/checkpoint-best.pth"
    supernet = "vit_b_16"
    dataset_path = "/dataset/imagenet-ilsvrc2012"
    device = "cuda"
    config = "/localdisk/sairamsu/vit_21k_search/huggingface_elastic_vit-iccv_workshop_work/configs/supernet_config.yml"
    supernet_config = config_parser(config)
    runner = ViTRunner(
        supernet=supernet,
        dataset_path=dataset_path,
        acc_predictor=None,
        macs_predictor=None,
        latency_predictor=None,
        params_predictor=None,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        total_batches=None,
        device=device,
    )
    # print(f"Top@1 Accuracy: {runner.validate_accuracy_imagenet(supernet_config)}\n")
    print(f"Latency (ms): {runner.measure_latency(supernet_config)}\n")
    print(f"MACs: {runner.validate_macs(supernet_config)}\n")


if __name__ == "__main__":
    main()
