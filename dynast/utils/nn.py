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
import time
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchprofile

from dynast.utils import log, measure_time


class AverageMeter(object):
    """Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


@measure_time
def validate_classification(
    model,
    data_loader,
    device='cpu',
):
    test_criterion = nn.CrossEntropyLoss()

    model = model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    total = len(data_loader)

    with torch.no_grad():
        for batch, (images, labels) in enumerate(data_loader):
            log.debug(
                "Validate #{}/{} {}".format(
                    batch + 1,
                    total,
                    {
                        "loss": losses.avg,
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "img_size": images.size(2),
                    },
                )
            )
            images, labels = images.to(device), labels.to(device)

            output = model(images)

            loss = test_criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

    return losses.avg, top1.avg, top5.avg


def get_parameters(
    model: nn.Module,
    device: str = 'cpu',
) -> int:
    model = copy.deepcopy(model)
    rm_bn_from_net(model)
    model = model.to(device)
    model = model.eval()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@measure_time
def get_macs(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: str = 'cpu',
) -> float:
    model = copy.deepcopy(model)
    rm_bn_from_net(model)
    model = model.to(device)
    model = model.eval()

    inputs = torch.randn(*input_size, device=device)
    macs = torchprofile.profile_macs(model, inputs)

    return macs


@measure_time
def reset_bn(  # TODO(Maciej) This should be renamed to `model_fine_tune` or `model_train` and a new method for calibration should be added
    model: nn.Module,
    num_samples: int,
    train_dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = 'cpu',
) -> None:
    model.train()
    model.to(device)

    batch_size = train_dataloader.batch_size

    if num_samples / batch_size > len(train_dataloader):
        log.warn("BN set stats: num of samples exceed the samples in loader. Using full loader")
    for i, (images, _) in enumerate(train_dataloader):
        log.debug("Calibrating BN statistics #{}/{}".format(i, num_samples // batch_size + 1))
        images = images.to(device)
        model(images)
        if i > num_samples / batch_size:
            log.info(f"Finishing setting bn stats using {num_samples} and batch size of {batch_size}")
            break

    if 'cuda' in str(device):
        log.debug('GPU mem peak usage: {} MB'.format(torch.cuda.max_memory_allocated() // 1024 // 1024))

    model.eval()


def rm_bn_from_net(
    net: nn.Module,
) -> None:
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    warmup_steps: int = 10,
    measure_steps: int = 50,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """Measure Torch model's latency.

    Returns
    -------
    `(mean latency; std latency)`
    """
    # TODO(macsz) Compare results with https://pytorch.org/tutorials/recipes/recipes/benchmark.html
    # TODO(macsz) Should also consider setting `omp_num_threads` here.
    times = []

    inputs = torch.randn(*input_size, device=device)
    model = model.eval()

    model = copy.deepcopy(model)
    rm_bn_from_net(model)
    model = model.to(device)

    if 'cuda' in str(device):
        torch.cuda.synchronize()

    log.debug('Warming up for {} steps...'.format(warmup_steps))
    for _ in range(warmup_steps):
        model(inputs)
    if 'cuda' in str(device):
        torch.cuda.synchronize()

    log.debug('Measuring latency for {} steps'.format(measure_steps))
    for _ in range(measure_steps):
        if 'cuda' in str(device):
            torch.cuda.synchronize()
        st = time.time()  # TODO(macsz) Use timeit instead
        model(inputs)
        if 'cuda' in str(device):
            torch.cuda.synchronize()
        ed = time.time()
        times.append(ed - st)

    # Convert to [ms] and round to 0.001
    latency_mean = np.round(np.mean(times) * 1e3, 3)
    latency_std = np.round(np.std(times) * 1e3, 3)

    return latency_mean, latency_std
