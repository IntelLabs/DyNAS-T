import functools as _functools
import json
import logging as _logging
import os
import subprocess
import time as _time

import numpy as np
import ofa  # TODO(Maciej) Remove OFA dependency
import onnxruntime
import pandas as pd
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count
from ofa.imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_codebase.run_manager import \
    ImagenetRunConfig  # TODO(Maciej) Remove OFA dependency
from rich.console import Console as _Console
from rich.logging import RichHandler as _RichHandler
from tqdm import tqdm

console = _Console()
_logging.basicConfig(
    level='INFO',
    format="%(message)s",
    datefmt='[%X]',
    handlers=[_RichHandler()],
)

log = _logging.getLogger('rich')


def measure_time(func):
    """ Decorator to measure elapsed time of a function call.

    Usage:

    ```
    @measure_time
    def foo(bar=2):
        return [i for i in range(bar)]

    print(foo())
    # Will print:
    # > Calling foo
    # [0, 1, 2]
    # > Finished foo in 0.0004 s
    ```
    """
    @_functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        log.info("> Calling {}".format(func.__name__))
        start_time = _time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = _time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        log.info('> Finished {} in {:.4f} s'.format(func.__name__, run_time))
        return value
    return wrapper_timer


def samples_to_batch_multiply(base_samples, batch_size):
    return (base_samples//batch_size+1)*batch_size


class AverageMeter(object):
    """
    Computes and stores the average and current value
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


def accuracy(output, target, topk=(1,)):
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
    epoch=0,
    is_test=True,
    run_str='',
    net=None,
    data_loader=None,
    no_logs=False,
    test_size=None,
    is_openvino=False,
    device='cpu',
    batch_size=128,
    workers=4
):
    ImagenetDataProvider.DEFAULT_PATH = '/datasets/imagenet-ilsvrc2012/'
    is_onnx = isinstance(net, onnxruntime.InferenceSession)
    test_criterion = nn.CrossEntropyLoss()
    run_config = ImagenetRunConfig(test_batch_size=batch_size, n_worker=workers)

    if (not is_openvino) and (not is_onnx):
        if (not isinstance(net, nn.DataParallel)):
            net = nn.DataParallel(net)
        net = net.eval()

    if data_loader is None:
        if is_test:
            data_loader = run_config.test_loader
        else:
            data_loader = run_config.valid_loader

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if test_size is not None:
        total = test_size
    else:
        total = len(data_loader)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    with torch.no_grad():
        with tqdm(total=total, desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)

                # if ofa.config.USE_MKLDNN:
                #     images = images.to_mkldnn()

                # compute output
                if is_onnx:
                    output = net.run([net.get_outputs()[0].name], {net.get_inputs()[0].name: to_numpy(images)})
                    output = torch.from_numpy(output[0]).to(device)
                elif is_openvino:
                    expected_batch_size = net.inputs['input'].shape[0]
                    img = to_numpy(images)
                    batch_size = len(img)

                    # openvino cannot handle dynamic batch sizes, so for
                    # the last batch of dataset, zero pad the batch size
                    if batch_size != expected_batch_size:
                        assert batch_size < expected_batch_size
                        npad = expected_batch_size - batch_size
                        img = np.pad(img, ((0, npad), (0, 0), (0, 0), (0, 0)), mode='constant')
                        img = img.copy()

                    output = net.infer(inputs={'input': img})
                    output = torch.Tensor(output['output'])[:batch_size]
                else:
                    output = net(images)

                if ofa.config.USE_MKLDNN:
                    output = output.to_dense()

                loss = test_criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))
                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                    'img_size': images.size(2),
                })
                t.update(1)
                if i > total:
                    break
    return losses.avg, top1.avg, top5.avg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@measure_time
def get_gflops(
    model,
    input_size=(1, 3, 224, 224),
    device='cpu',
):
    input = torch.randn(*input_size, device=device)
    flops = FlopCountAnalysis(model, input)
    flop_batch_size = input_size[0]
    gflops = flops.total()/(flop_batch_size*10**9)
    log.info('Model\'s GFLOPs: {}'.format(gflops))
    return gflops


def get_hostname():
    return os.getenv('HOSTNAME', os.getenv('HOST', 'unnamed-host'))


def get_cores(num_cores: int):
    """ For a given number of cores, returns the core IDs that should be used.

    This script prioritizes using cores from the same socket first. e.g. for a
    two socket CLX 8280 system, that means using cores: 0-27, 56-83, 28-55, 84-111
    in that order, since [0-27, 56-83] belong to the same socket.
    """
    cmd = ['lscpu', '--json', '--extended']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, code = p.communicate()
    cpu = json.loads(out)

    df = pd.DataFrame.from_dict(cpu['cpus'])
    for key in ['cpu', 'node', 'socket', 'core']:
        df[key] = df[key].astype(int)

    df = df.sort_values(['node', 'socket', 'cpu'])
    cores = df['cpu'].to_list()[:num_cores]
    cores = [str(c) for c in cores]
    return ','.join(cores)
