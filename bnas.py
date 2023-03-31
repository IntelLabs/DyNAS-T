import argparse
import logging
import random
import sys
from typing import Dict, List

import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from nncf import set_log_level
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import SubnetConfig
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_from_state
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.model_creation import create_nncf_network
from torch import nn

from dynast.dynast_manager import DyNAS
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_utils import load_base_model, load_config
from dynast.utils import log, set_logger
from dynast.utils.datasets import CIFAR10
from dynast.utils.nn import get_macs, reset_bn, validate_classification

set_log_level(logging.ERROR)
set_logger(logging.INFO)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

NUM_EVALS = 1


class BootstrapNAS:
    """This class ideally would be placed in the nncf package, but for now it is here to avoid
    too many unnecessary steps to run the example.

    Original import:
    `from nncf.experimental.torch.nas.bootstrapNAS.bootstrapNAS import BootstrapNAS`

    Few of the implemented methods are not needed. We can currate them later.
    """

    def __init__(self, model: torch.nn.Module, nncf_config: Dict):
        nncf_network = create_nncf_network(model, nncf_config)

        compression_state = torch.load(nncf_config.supernet_path, map_location=torch.device(nncf_config.device))
        self._model, self._elasticity_ctrl = resume_compression_from_state(nncf_network, compression_state)
        model_weights = torch.load(nncf_config.supernet_weights, map_location=torch.device(nncf_config.device))

        load_state(model, model_weights, is_resume=True)
        self.nncf_config = nncf_config

    def get_search_space(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        active_handlers = {
            dim: m_handler._handlers[dim] for dim in m_handler._handlers if m_handler._is_handler_enabled_map[dim]
        }
        space = {}
        for handler_id, handler in active_handlers.items():
            space[handler_id.value] = handler.get_search_space()
        return space

    def eval_subnet(self, config, eval_fn, **kwargs):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        m_handler.activate_subnet_for_config(
            # m_handler.get_config_from_pymoo(config)
            config
        )
        print(kwargs)
        return eval_fn(self._model, **kwargs)

    def get_active_subnet(self):
        return self._model

    def get_active_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_active_config()

    def get_random_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_random_config()

    def get_minimum_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_minimum_config()

    def get_maximum_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_maximum_config()

    def get_available_elasticity_dims(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_available_elasticity_dims()

    def activate_subnet_for_config(self, config):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        m_handler.activate_subnet_for_config(config)

    def activate_subnet_for_pymoo_config(self, config):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        config = m_handler.get_config_from_pymoo(config)
        m_handler.activate_subnet_for_config(config)

    def get_config_from_pymoo(self, x: List) -> SubnetConfig:
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_config_from_pymoo(x)

    def eval_subnet_with_design_vars(self, pymoo_config, eval_func):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler

        m_handler.activate_subnet_for_config(m_handler.get_config_from_pymoo(pymoo_config))

        return eval_func(self.get_active_subnet())

    def get_macs_for_active_config(self):
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return get_macs(
            model=self.get_active_subnet(),
            input_size=(1, 3, 32, 32),
            device=self.nncf_config.device,
        )


def get_nas_argument_parser():
    """Returns the argument parser for the NAS example. Just for ease-of-use. Can be removed later."""
    parser = argparse.ArgumentParser(description="TODO")

    parser.add_argument(
        "--supernet-path",
        default="/store/code/bootstrapnas/Hardware-Aware-Automated-Machine-Learning/models/supernets/cifar10/resnet50/elasticity.pth",
        type=str,
        help="Path of elasticity state",
    )

    parser.add_argument(
        "--supernet-weights",
        default="/store/code/bootstrapnas/Hardware-Aware-Automated-Machine-Learning/models/supernets/cifar10/resnet50/supernet_weights.pth",
        type=str,
        help="Path to weights of trained super-network",
    )

    parser.add_argument("--log-dir", type=str, default="runs", help="The directory where logs" " are saved.")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=4000, help="Number of samples used to reset batch norm")
    parser.add_argument("--num_evals", type=int, default=1, help="Number of configurations to sample")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--dataset_path", type=str, default="/tmp/cifar10")
    parser.add_argument("--out_fn", type=str, default="out_1.csv")
    return parser


def validate(model, test_dataloader, train_dataloader, config, bn=True):
    """Ignore this function for the DyNAS-T example."""
    model.eval()

    if bn:
        log.info("Adjusting BN...")
        adapt_bn(model=model, train_dataloader=train_dataloader, config=config)

    loss, top1, top5 = validate_classification(
        model=model,
        data_loader=test_dataloader,
        device=config.device,
    )
    return top1


def write(line, scratch=False, fn="out.csv"):
    """Ignore this function for the DyNAS-T example."""
    with open(fn, "w" if scratch else "a") as f:
        f.write(f"{line}\n")


def adapt_bn(model, train_dataloader, config):
    """Ignore this function for the DyNAS-T example."""
    reset_bn(
        model=model,
        num_samples=200,
        train_dataloader=train_dataloader,
        device=config.device,
    )


def main(argv):
    random.seed(42)
    parser = get_nas_argument_parser()
    args = parser.parse_args(argv)

    config = load_config(args)
    model = load_base_model(config)

    log.info("Bootstrapping model...")
    bootstrapNAS = BootstrapNAS(model, config)

    log.debug("Search space: {}".format(bootstrapNAS.get_search_space()))
    log.debug(f"Available elasticity: {bootstrapNAS.get_available_elasticity_dims()}")
    log.debug(f"Min config: {bootstrapNAS.get_minimum_config()}")
    log.debug(f"Max config: {bootstrapNAS.get_maximum_config()}")

    if True:
        search_tactic = 'random'
        if 'random' in search_tactic:
            dynast_config = {
                'search_tactic': 'random',
                'results_path': 'bootstrapnas_resnet50_cifar10_random.csv',
                'num_evals': 1,
                'population': 1,
            }
        elif 'linas' in search_tactic:
            dynast_config = {
                'search_tactic': 'linas',
                'results_path': 'bootstrapnas_resnet50_cifar10_linas.csv',
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
            device=config.device,
            verbose=False,
            **dynast_config,
        )
        dynas.search()
    else:
        testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

        trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    if False:
        acc = validate(model=model, test_dataloader=testloader, train_dataloader=trainloader, config=config, bn=False)
        macs = get_macs(model, input_size=config.input_info.sample_size, device=config.device)
        log.info(f"model MACs: {macs}, top1: {acc}")
        write(f"{macs}, {acc}", scratch=True, fn=args.out_fn)
    if False:
        bootstrapNAS.activate_subnet_for_config(bootstrapNAS.get_minimum_config())
        subnet_min = bootstrapNAS.get_active_subnet()
        acc_min = validate(model=subnet_min, test_dataloader=testloader, train_dataloader=trainloader, config=config)
        macs_min = get_macs(subnet_min, input_size=config.input_info.sample_size, device=config.device)
        log.info(f"Min MACs: {macs_min}, top1: {acc_min}")
        write(f"{macs_min}, {acc_min}", fn=args.out_fn)

    if False:
        bootstrapNAS.activate_subnet_for_config(bootstrapNAS.get_maximum_config())
        subnet_max = bootstrapNAS.get_active_subnet()
        acc_max = validate(model=subnet_max, test_dataloader=testloader, train_dataloader=trainloader, config=config)
        macs_max = get_macs(subnet_max, input_size=config.input_info.sample_size, device=config.device)
        log.info(f"Max MACs: {macs_max}, top1: {acc_max}")
        write(f"{macs_max}, {acc_max}", fn=args.out_fn)

    if False:
        for _ in tqdm.tqdm(range(args.num_evals)):
            bootstrapNAS.activate_subnet_for_config(bootstrapNAS.get_random_config())
            subnet = bootstrapNAS.get_active_subnet()
            acc = validate(model=subnet, test_dataloader=testloader, train_dataloader=trainloader, config=config)
            macs = get_macs(subnet, input_size=config.input_info.sample_size, device=config.device)
            log.info(f"Random MACs: {macs}, top1: {acc}")
            write(f"{macs}, {acc}", fn=args.out_fn)

    if False:
        config = bootstrapNAS.get_random_config()
        results = bootstrapNAS.eval_subnet(
            config=config,
            eval_fn=validate_classification,
            data_loader=CIFAR10.validation_dataloader(256),
            device="cuda",
        )

    if False:
        results = validate(CIFAR10.validation_dataloader(256), model, nn.CrossEntropyLoss(), config)


if __name__ == "__main__":
    main(sys.argv[1:])
