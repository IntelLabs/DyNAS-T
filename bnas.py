import logging
import random

import torchvision.transforms as transforms
from nncf import set_log_level
from nncf.experimental.torch.nas.bootstrapNAS.search.supernet import SuperNetwork

from dynast.dynast_manager import DyNAS
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_utils import load_base_model, load_config
from dynast.utils import log, set_logger

set_log_level(logging.ERROR)
set_logger(logging.INFO)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def main():
    random.seed(42)

    search_name = "dynast_bnas_external"
    log_dir = "runs"
    config_path = "/store/code/bootstrapnas/Hardware-Aware-Automated-Machine-Learning/models/supernets/cifar10/resnet50/config.json"
    supernet_path = "/store/code/bootstrapnas/Hardware-Aware-Automated-Machine-Learning/models/supernets/cifar10/resnet50/elasticity.pth"
    supernet_weights = "/store/code/bootstrapnas/Hardware-Aware-Automated-Machine-Learning/models/supernets/cifar10/resnet50/supernet_weights.pth"
    dataset = "cifar10"
    batch_size = 256
    fp32_pth_url = "/store/code/bootstrapnas/Hardware-Aware-Automated-Machine-Learning/models/pretrained/resnet50.pt"

    nncf_config = load_config(
        config_path=config_path,
        log_dir=log_dir,
        dataset=dataset,
        batch_size=batch_size,
        search_name=search_name,
    )
    model = load_base_model(fp32_pth_url, nncf_config.device)

    log.info("Bootstrapping model...")
    bootstrapNAS = SuperNetwork.from_checkpoint(
        model=model,
        nncf_config=nncf_config,
        supernet_path=supernet_path,
        supernet_weights=supernet_weights,
    )

    if True:
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
        dynas = DyNAS(
            supernet='bootstrapnas_image_classification',
            optimization_metrics=['accuracy_top1', 'macs'],
            measurements=['accuracy_top1', 'macs'],
            batch_size=256,
            dataset_path='/tmp/cifar10',
            bootstrapnas=bootstrapNAS,  # This is the only new param that has to be passed
            device=nncf_config.device,
            verbose=False,
            **dynast_config,
        )
        dynas.search()


if __name__ == "__main__":
    main()
