import argparse
import json
import logging
import random
import sys
from typing import List

import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from addict import Dict
from nncf import set_log_level
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import SubnetConfig
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_from_state
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.model_creation import create_nncf_network
from torch import nn

from bootstrapnas_utils import resnet50_cifar10
from dynast.utils import log, set_logger
from dynast.utils.datasets import CIFAR10
from dynast.utils.nn import get_macs, reset_bn, validate_classification

set_log_level(logging.ERROR)
set_logger(logging.INFO)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

NUM_EVALS = 1


class BootstrapNAS:
    def __init__(self, model, nncf_config, supernet_path, supernet_weights):
        nncf_network = create_nncf_network(model, nncf_config)

        compression_state = torch.load(supernet_path, map_location=torch.device(nncf_config.device))
        self._model, self._elasticity_ctrl = resume_compression_from_state(nncf_network, compression_state)
        model_weights = torch.load(supernet_weights, map_location=torch.device(nncf_config.device))

        load_state(model, model_weights, is_resume=True)

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
            m_handler.get_config_from_pymoo(config)
            # config
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

    def get_config_from_pymoo(self, x: List) -> SubnetConfig:
        m_handler = self._elasticity_ctrl.multi_elasticity_handler
        return m_handler.get_config_from_pymoo(x)


def get_nas_argument_parser():
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
    parser.add_argument("--out_fn", type=str, default="out_1.csv")
    return parser


def load_base_model(config):
    fp32_pth_url = "/store/code/bootstrapnas/Hardware-Aware-Automated-Machine-Learning/models/pretrained/resnet50.pt"
    log.info(f"Loading base model {fp32_pth_url}...")
    model = resnet50_cifar10()
    state_dict = torch.load(fp32_pth_url)
    model.load_state_dict(state_dict)
    model.to(config.device)
    return model


def load_config(args):
    config_path = "/store/code/bootstrapnas/Hardware-Aware-Automated-Machine-Learning/models/supernets/cifar10/resnet50/config.json"
    log.info(f"Loading config {config_path}...")
    with open(config_path) as f:
        config = Dict(json.load(f))
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {config.device}")
    config.log_dir = args.log_dir
    config.checkpoint_save_dir = config.log_dir
    config.supernet_path = args.supernet_path
    config.supernet_weights = args.supernet_weights
    config.batch_size = args.batch_size
    config.dataset = args.dataset
    config.name = "dynast_bnas_external"
    return config


# from examples.torch.classification.main import validate


def validate(model, config, bn=True):
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    if bn:
        log.info("Adjusting BN...")
        adapt_bn(model, config)
    with torch.no_grad():
        for data in tqdm.tqdm(testloader):
            images, labels = data
            images, labels = images.to(config.device), labels.to(config.device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    top1 = 100 * correct / total
    log.info(f"Accuracy of the network on the 10000 test images: {top1} %")
    return top1


def write(line, scratch=False, fn="out.csv"):
    with open(fn, "w" if scratch else "a") as f:
        f.write(f"{line}\n")


def adapt_bn(model, config):
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    reset_bn(
        model=model,
        num_samples=200,
        train_dataloader=trainloader,
        device=config.device,
    )


def main(argv):
    random.seed(42)
    parser = get_nas_argument_parser()
    args = parser.parse_args(argv)

    config = load_config(args)
    model = load_base_model(config)

    acc = validate(model, config, False)
    macs = get_macs(model, input_size=config.input_info.sample_size, device=config.device)
    log.info(f"model MACs: {macs}, top1: {acc}")
    write(f"{macs}, {acc}", scratch=True, fn=args.out_fn)

    log.info("Bootstrapping model...")
    bootstrapNAS = BootstrapNAS(model, config, config.supernet_path, config.supernet_weights)

    log.info("Search space: {}".format(bootstrapNAS.get_search_space()))

    log.info(f"Available elasticity: {bootstrapNAS.get_available_elasticity_dims()}")
    log.info(f"Min config: {bootstrapNAS.get_minimum_config()}")
    log.info(f"Max config: {bootstrapNAS.get_maximum_config()}")

    if False:
        bootstrapNAS.activate_subnet_for_config(bootstrapNAS.get_minimum_config())
        subnet_min = bootstrapNAS.get_active_subnet()
        acc_min = validate(subnet_min, config)
        macs_min = get_macs(subnet_min, input_size=config.input_info.sample_size, device=config.device)
        log.info(f"Min MACs: {macs_min}, top1: {acc_min}")
        write(f"{macs_min}, {acc_min}", fn=args.out_fn)

    if False:
        bootstrapNAS.activate_subnet_for_config(bootstrapNAS.get_maximum_config())
        subnet_max = bootstrapNAS.get_active_subnet()
        acc_max = validate(subnet_max, config)
        macs_max = get_macs(subnet_max, input_size=config.input_info.sample_size, device=config.device)
        log.info(f"Max MACs: {macs_max}, top1: {acc_max}")
        write(f"{macs_max}, {acc_max}", fn=args.out_fn)

    for _ in tqdm.tqdm(range(args.num_evals)):
        bootstrapNAS.activate_subnet_for_config(bootstrapNAS.get_random_config())
        subnet = bootstrapNAS.get_active_subnet()
        acc = validate(subnet, config)
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

    # print(results)


if __name__ == "__main__":
    main(sys.argv[1:])