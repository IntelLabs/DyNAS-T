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
from dynast.supernetwork.image_classification.vit_quantized.vit_quantized_interface import ViTQuantizedRunner
from dynast.utils import log, set_logger
from dynast.utils.datasets import ImageNet
from dynast.utils.nn import validate_classification

set_logger(logging.DEBUG)

warnings.filterwarnings("ignore")


def config_parser(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    batch_size = 128
    checkpoint_path = (
        "/nfs/site/home/mszankin/store/code/huggingface_elastic_vit/results/elastic_vit_lr-2e-2_ST/checkpoint-best.pth"
    )
    supernet = "vit_b_16"
    dataset_path = "/localdisk/dataset/imagenet/"
    # device = "cuda:0"
    device="cpu"
    supernet_config = {
        "embedding_size": 512,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
        "intermediate_sizes": [3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072],
        'q_bits': 74*[32],
    }
    runner = ViTQuantizedRunner(
        supernet=supernet,
        dataset_path=dataset_path,
        acc_predictor=None,
        # macs_predictor=None,
        model_size_predictor=None,
        latency_predictor=None,
        params_predictor=None,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        device=device,
        test_fraction=0.2,
    )

    q_model = runner.quantize_subnet(
        subnet_config=supernet_config,
        qbit_list=supernet_config['q_bits'],
    )
    print(f"Top@1 Accuracy: {runner.validate_accuracy_imagenet(q_model)}\n")
    # print(f"Latency (ms): {runner.measure_latency(supernet_config)}\n")
    # print(f"MACs: {runner.validate_macs(supernet_config)}\n")
    print(f"Model Size: {runner.measure_model_size(q_model)}\n")


if __name__ == "__main__":
    main()
