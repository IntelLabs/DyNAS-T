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


def main():
    log.info('$PYTHONPATH: {}'.format(sys.path))
    random.seed(42)

    haaml_path = Path(
        "/localdisk/maciej/code/dynast_bootstrapnas_integration/hardware_aware_automated_machine_learning"
    )

    config_path = haaml_path / "models/supernets/cifar10/resnet50/config.json"
    supernet_path = haaml_path / "models/supernets/cifar10/resnet50/elasticity.pth"
    supernet_weights = haaml_path / "models/supernets/cifar10/resnet50/supernet_weights.pth"
    fp32_pth_url = haaml_path / "models/pretrained/resnet50.pt"

    log.info(f"Loading config {config_path}...")
    with open(config_path) as f:
        nncf_config = Dict(json.load(f))
    nncf_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO(macsz) Update this.
    log.info(f"Using device: {nncf_config.device}")
    nncf_config.log_dir = "runs"
    nncf_config.checkpoint_save_dir = nncf_config.log_dir
    nncf_config.batch_size = 256
    nncf_config.dataset = "cifar10"
    nncf_config.name = "dynast_bnas_external"

    log.info(f"Loading base model {fp32_pth_url}...")
    model = resnet50_cifar10()
    state_dict = torch.load(fp32_pth_url)
    model.load_state_dict(state_dict)
    model.to(nncf_config.device)

    log.info("Bootstrapping model...")
    bootstrapNAS = SuperNetwork.from_checkpoint(
        model=model,
        nncf_config=nncf_config,
        supernet_path=supernet_path,
        supernet_weights=supernet_weights,
    )

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

    dynast_config['supernet'] = 'bootstrapnas_image_classification'
    dynast_config['test_fraction'] = 1.0
    dynast_config['optimization_metrics'] = ['accuracy_top1', 'macs']
    dynast_config['measurements'] = ['accuracy_top1', 'macs']
    dynast_config['batch_size'] = nncf_config.batch_size
    dynast_config['dataset_path'] = '/tmp/cifar10'
    dynast_config['bootstrapnas'] = bootstrapNAS  # This is the only new param that has to be passed
    dynast_config['device'] = nncf_config.device
    dynast_config['verbose'] = False

    dynas = DyNAS(**dynast_config)
    dynas.search()


if __name__ == "__main__":
    main()
