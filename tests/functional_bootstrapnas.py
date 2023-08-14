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

import logging
import random
import sys
from pathlib import Path

import torch
from addict import Dict
from examples.torch.classification.main import create_data_loaders, create_datasets
from examples.torch.common.execution import set_seed
from examples.torch.common.models.classification.resnet_cifar10 import resnet50_cifar10
from nncf import set_log_level
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.config.config import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS.search.supernet import TrainedSuperNet
from nncf.torch.initialization import wrap_dataloader_for_init

from dynast.dynast_manager import DyNAS
from dynast.utils import log, set_logger
from dynast.utils.nn import get_macs, validate_classification

set_log_level(logging.ERROR)
set_logger(logging.INFO)

HAAML_PATH = Path("/localdisk/maciej/code/dynast_bootstrapnas_integration/hardware_aware_automated_machine_learning")
CONFIG_FILE_PATH = HAAML_PATH / "models/supernets/cifar10/resnet50/config.json"
SUPERNET_PATH = HAAML_PATH / "models/supernets/cifar10/resnet50/elasticity.pth"
SUPERNET_WEIGHTS = HAAML_PATH / "models/supernets/cifar10/resnet50/supernet_weights.pth"
BASE_MODEL_PATH = HAAML_PATH / "models/pretrained/resnet50.pt"


def create_nncf_config(config_file_path: str):
    log.info(f"Loading config {config_file_path}...")
    nncf_config = NNCFConfig.from_json(config_file_path)
    nncf_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO(macsz) Update this.
    log.info(f"Using device: {nncf_config.device}")
    nncf_config.log_dir = "runs"
    nncf_config.checkpoint_save_dir = nncf_config.log_dir
    nncf_config.batch_size = 256
    nncf_config.batch_size_val = 256
    nncf_config.batch_size_init = 256
    nncf_config.name = "dynast_bnas_external"
    nncf_config.dataset = "cifar10"
    nncf_config.dataset_dir = "/tmp/cifar10"
    nncf_config.workers = 4
    nncf_config.execution_mode = "single_gpu"
    nncf_config.distributed = False
    nncf_config.seed = 42

    set_seed(nncf_config)

    return nncf_config


def create_dynast_config(nncf_config: Dict, bootstrapnas_supernetwork: TrainedSuperNet):
    search_tactic = 'linas'
    if 'random' in search_tactic:
        dynast_config = {
            'search_tactic': 'random',
            'results_path': 'bootstrapnas_resnet50_cifar10_random_test_1.csv',
            'num_evals': 500,
            'population': 500,
        }
    elif 'linas' in search_tactic:
        dynast_config = {
            'search_tactic': 'linas',
            'results_path': 'bootstrapnas_resnet50_cifar10_linas_test_1.csv',
            'num_evals': 25,
            'population': 5,
        }

    dynast_config.update(
        {
            'seed': nncf_config.seed,
            'supernet': 'bootstrapnas_image_classification',
            'test_fraction': 1.0,
            'optimization_metrics': ['accuracy_top1', 'macs'],
            'measurements': ['accuracy_top1', 'macs'],
            'batch_size': nncf_config.batch_size,
            'dataset_path': nncf_config.dataset_dir,
            'bootstrapnas_supernetwork': bootstrapnas_supernetwork,  # This is the only new param that has to be passed
            'device': nncf_config.device,
            'verbose': False,
        }
    )
    return dynast_config


def load_model(nncf_config):
    log.info(f"Loading base model {BASE_MODEL_PATH}...")
    model = resnet50_cifar10()
    state_dict = torch.load(BASE_MODEL_PATH)
    model.load_state_dict(state_dict)
    model.to(nncf_config.device)
    return model


def prepare_dataloaders(nncf_config):
    train_dataset, val_dataset = create_datasets(nncf_config)
    train_loader, _, val_loader, _ = create_data_loaders(nncf_config, train_dataset, val_dataset)
    return train_loader, val_loader


def main():
    log.info('$PYTHONPATH: {}'.format(sys.path))
    random.seed(42)

    nncf_config = create_nncf_config(CONFIG_FILE_PATH)

    train_loader, val_loader = prepare_dataloaders(nncf_config)

    bn_adapt_args = BNAdaptationInitArgs(data_loader=wrap_dataloader_for_init(train_loader), device=nncf_config.device)
    nncf_config.register_extra_structs([bn_adapt_args])

    bn_adapt_algo_kwargs = {
        'data_loader': train_loader,
        'num_bn_adaptation_samples': 6000,
        'device': 'cuda',
    }
    bn_adaptation = BatchnormAdaptationAlgorithm(**bn_adapt_algo_kwargs)

    model = load_model(nncf_config)

    log.info("Bootstrapping model...")
    bootstrapnas_supernetwork = TrainedSuperNet.from_checkpoint(
        model=model,
        nncf_config=nncf_config,
        supernet_elasticity_path=SUPERNET_PATH,
        supernet_weights_path=SUPERNET_WEIGHTS,
    )

    dynast_config = create_dynast_config(nncf_config, bootstrapnas_supernetwork)

    def accuracy_top1_fn(_model):
        bn_adaptation.run(_model)

        losses, top1, top5 = validate_classification(
            model=_model,
            data_loader=val_loader,
            device=nncf_config.device,
        )
        return top1

    dynast_config['metric_eval_fns'] = {
        'accuracy_top1': accuracy_top1_fn,
    }

    dynas = DyNAS(**dynast_config)
    dynas.search()


if __name__ == "__main__":
    main()
