import argparse
import json
import logging
import random
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from addict import Dict
from nncf import set_log_level
from torch import nn

from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_interface import BootstrapNAS
from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_utils import resnet50_cifar10
from dynast.utils import log, set_logger
from dynast.utils.datasets import CIFAR10
from dynast.utils.nn import get_macs, reset_bn, validate_classification

set_log_level(logging.ERROR)
set_logger(logging.INFO)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

NUM_EVALS = 1


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


def validate(model, test_dataloader, train_dataloader, config, bn=True):
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
    with open(fn, "w" if scratch else "a") as f:
        f.write(f"{line}\n")


def adapt_bn(model, train_dataloader, config):
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

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    acc = validate(model=model, test_dataloader=testloader, train_dataloader=trainloader, config=config, bn=False)
    macs = get_macs(model, input_size=config.input_info.sample_size, device=config.device)
    log.info(f"model MACs: {macs}, top1: {acc}")
    write(f"{macs}, {acc}", scratch=True, fn=args.out_fn)

    log.info("Bootstrapping model...")
    bootstrapNAS = BootstrapNAS(model, config)

    log.info("Search space: {}".format(bootstrapNAS.get_search_space()))

    log.info(f"Available elasticity: {bootstrapNAS.get_available_elasticity_dims()}")
    log.info(f"Min config: {bootstrapNAS.get_minimum_config()}")
    log.info(f"Max config: {bootstrapNAS.get_maximum_config()}")

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

    # print(results)


if __name__ == "__main__":
    main(sys.argv[1:])
