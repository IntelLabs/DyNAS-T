import json
import logging
import random
import sys
from pathlib import Path

import torch
from addict import Dict
from examples.torch.common.models.classification.resnet_cifar10 import resnet50_cifar10
from nncf import set_log_level
from nncf.experimental.torch.nas.bootstrapNAS.search.supernet import SuperNetwork

from dynast.dynast_manager import DyNAS
from dynast.utils import log, set_logger

set_log_level(logging.ERROR)
set_logger(logging.INFO)

HAAML_PATH = Path("/localdisk/maciej/code/dynast_bootstrapnas_integration/hardware_aware_automated_machine_learning")
CONFIG_FILE_PATH = HAAML_PATH / "models/supernets/cifar10/resnet50/config.json"
SUPERNET_PATH = HAAML_PATH / "models/supernets/cifar10/resnet50/elasticity.pth"
SUPERNET_WEIGHTS = HAAML_PATH / "models/supernets/cifar10/resnet50/supernet_weights.pth"
BASE_MODEL_PATH = HAAML_PATH / "models/pretrained/resnet50.pt"


def create_nncf_config(config_file_path: str):
    log.info(f"Loading config {config_file_path}...")
    with open(config_file_path) as f:
        nncf_config = Dict(json.load(f))
    nncf_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO(macsz) Update this.
    log.info(f"Using device: {nncf_config.device}")
    nncf_config.log_dir = "runs"
    nncf_config.checkpoint_save_dir = nncf_config.log_dir
    nncf_config.batch_size = 256
    nncf_config.dataset = "cifar10"
    nncf_config.name = "dynast_bnas_external"

    return nncf_config


def create_dynast_config(nncf_config: Dict, bootstrapNAS: SuperNetwork):
    search_tactic = 'random'
    if 'random' in search_tactic:
        dynast_config = {
            'search_tactic': 'random',
            'results_path': 'bootstrapnas_resnet50_cifar10_random_test.csv',
            'num_evals': 1,
            'population': 1,
        }
    elif 'linas' in search_tactic:
        dynast_config = {
            'search_tactic': 'linas',
            'results_path': 'bootstrapnas_resnet50_cifar10_linas_test.csv',
            'num_evals': 250,
            'population': 50,
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
            'bootstrapnas_supernetwork': bootstrapNAS,  # This is the only new param that has to be passed
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


def main():
    log.info('$PYTHONPATH: {}'.format(sys.path))
    random.seed(42)

    nncf_config = create_nncf_config(CONFIG_FILE_PATH)

    model = load_model(nncf_config)

    log.info("Bootstrapping model...")
    bootstrapNAS = SuperNetwork.from_checkpoint(
        model=model,
        nncf_config=nncf_config,
        supernet_path=SUPERNET_PATH,
        supernet_weights=SUPERNET_WEIGHTS,
    )

    dynast_config = create_dynast_config(nncf_config, bootstrapnas_supernetwork)

    dynas = DyNAS(**dynast_config)
    dynas.search()


if __name__ == "__main__":
    main()
