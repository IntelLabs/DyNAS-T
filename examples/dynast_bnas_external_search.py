import argparse
import datetime
import logging
import os
import sys

from addict import Dict
from torch import nn

from dynast.utils import log, measure_time
from dynast.utils.datasets import CIFAR10
from dynast.utils.nn import reset_bn, validate_classification

try:
    from cifar10_models.mobilenetv2 import mobilenet_v2
except ImportError as ie:
    log.error('{exception}. To fix it please clone \'https://github.com/huyvnphan/PyTorch_CIFAR10\' add it to '
              '\'$PYTHONPATH\' and get model weights using instructions provided there.'.format(exception=ie))
    exit()

from nncf.common.utils.logger import set_log_level
from nncf.experimental.torch.nas.bootstrapNAS.bootstrapNAS import BootstrapNAS

set_log_level(logging.ERROR)


def configure_paths(config):
    d = datetime.datetime.now()
    run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
    config.log_dir = os.path.join(config.log_dir, "{}/{}".format(config.name, run_id))
    os.makedirs(config.log_dir)

    if config.checkpoint_save_dir is None:
        config.checkpoint_save_dir = config.log_dir

    # create aux dirs
    config.intermediate_checkpoints_path = config.log_dir + '/intermediate_checkpoints'
    os.makedirs(config.intermediate_checkpoints_path)
    os.makedirs(config.checkpoint_save_dir, exist_ok=True)


def get_nas_argument_parser():
    parser = argparse.ArgumentParser(description='TODO')

    parser.add_argument('--supernet-path', required=True, type=str,
                        help='Path of elasticity state')

    parser.add_argument('--supernet-weights', required=True, type=str,
                        help='Path to weights of trained super-network')

    parser.add_argument('--log-dir', type=str, default='runs', help='The directory where logs'
                        ' are saved.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    return parser


@measure_time
def validate_subnet(subnet: nn.Module):
    train_dataloader = CIFAR10.train_dataloader(batch_size=128)

    reset_bn(
        model=subnet,
        train_dataloader=train_dataloader,
        num_samples=4000,
        batch_size=128,
        device='cuda',
    )

    loss, top1, top5 = validate_classification(
        net=subnet,
        data_loader=CIFAR10.validation_dataloader(batch_size=128),
        no_logs=False,
        is_openvino=False,
        device='cuda',
        batch_size=128,
        workers=4,
    )

    return loss, top1, top5


def main(argv):
    parser = get_nas_argument_parser()
    args = parser.parse_args(argv)
    config = Dict()
    config.name = 'dynast_bnas_external'
    # TODO(Maciej) What's the significance of `config.dataset`? Changing it does not seem to do anything.
    config.dataset = args.dataset
    config.log_dir = args.log_dir
    config.checkpoint_save_dir = config.log_dir
    config.supernet_path = args.supernet_path
    config.supernet_weights = args.supernet_weights

    configure_paths(config)

    log.info('Loading model...')
    model = mobilenet_v2(pretrained=True)

    log.info('Bootstrapping model...')
    bootstrapNAS = BootstrapNAS(model, config, config.supernet_path, config.supernet_weights)
    log.info('Constructing search space...')
    search_space = bootstrapNAS.get_search_space()
    log.info('Search Space: {search_space}'.format(search_space=search_space))

    loss, top1, top5 = bootstrapNAS.eval_subnet(
        [0, 0, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
        validate_subnet,  # TODO(Maciej): How to pass additional params?
    )
    log.info('loss {loss} top1 {top1} top5 {top5}'.format(loss=loss, top1=top1, top5=top5))
    # TODO Do some magic with DyNAS-T!


if __name__ == '__main__':
    main(sys.argv[1:])
