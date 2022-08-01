import time
from typing import Tuple, Union

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from dynast.utils import log, measure_time


class AverageMeter(object):
    """ Computes and stores the average and current value
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


def accuracy(
    output,
    target,
    topk=(1,)
):
    """ Computes the accuracy over the k top predictions for the specified values of k
    """
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
    epoch=0,
    test_size=None,
    is_openvino=False,
    use_mkldnn=False,
    device='cpu',
    batch_size=128,
):
    # NOTE(macsz): if `use_mkldnn` is set to True and model is an OFA submodel,
    # please refer to HANDI OFA and follow MKLDNN instructions there.

    is_onnx = isinstance(model, onnxruntime.InferenceSession)
    test_criterion = nn.CrossEntropyLoss()

    if (not is_openvino) and (not is_onnx):
        if (not isinstance(model, nn.DataParallel)):
            model = nn.DataParallel(model)
        model = model.eval()

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
        for i, (images, labels) in enumerate(data_loader):
            epoch += 1
            log.debug(
                "Validate #{}/{} {}".format(
                    epoch,
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

            if use_mkldnn:
                images = images.to_mkldnn()

            # compute output
            if is_onnx:
                output = model.run(
                    [model.get_outputs()[0].name],
                    {model.get_inputs()[0].name: to_numpy(images)},
                )
                output = torch.from_numpy(output[0]).to(device)
            elif is_openvino:
                expected_batch_size = model.inputs["input"].shape[0]
                img = to_numpy(images)
                batch_size = len(img)

                # openvino cannot handle dynamic batch sizes, so for
                # the last batch of dataset, zero pad the batch size
                if batch_size != expected_batch_size:
                    assert (
                        batch_size < expected_batch_size
                    ), "Assert batch_size:{} < expected_batch_size:{}".format(
                        batch_size, expected_batch_size
                    )
                    npad = expected_batch_size - batch_size
                    img = np.pad(
                        img, ((0, npad), (0, 0), (0, 0), (0, 0)), mode="constant"
                    )
                    img = img.copy()

                output = model.infer(inputs={"input": img})
                output = torch.Tensor(output["output"])[:batch_size]
            else:
                output = model(images)

            if use_mkldnn:
                output = output.to_dense()

            loss = test_criterion(output, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
            if i > total:
                break
    return losses.avg, top1.avg, top5.avg


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@measure_time
def get_gflops(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: str = 'cpu',
) -> float:
    model.eval()
    model = model.to(device)
    input = torch.randn(*input_size, device=device)
    flops = FlopCountAnalysis(model, input)
    flop_batch_size = input_size[0]
    gflops = flops.total()/(flop_batch_size*10**9)
    log.info('Model\'s GFLOPs: {}'.format(gflops))
    return gflops


@measure_time
def reset_bn(  # TODO(Maciej) This can be reupruposed/merged with training method - same thing.
    model: nn.Module,
    num_samples: int,
    batch_size: int,
    train_dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = 'cpu',
) -> None:
    model.train()
    model.to(device)
    if num_samples / batch_size > len(train_dataloader):
        log.warn(
            "BN set stats: num of samples exceed the samples in loader. Using full loader"
        )
    for i, (images, _) in enumerate(train_dataloader):
        log.debug(
            "Calibrating BN statistics #{}/{}".format(i, num_samples // batch_size + 1)
        )
        images = images.to(device)
        model(images)
        if i > num_samples / batch_size:
            log.info(
                f"Finishing setting bn stats using {num_samples} and batch size of {batch_size}"
            )
            break

    if 'cuda' in str(device):
        log.debug('GPU mem peak usage: {} MB'.format(torch.cuda.max_memory_allocated()//1024//1024))

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
    times = []

    inputs = torch.randn(*input_size, device=device)
    model = model.eval()
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
        st = time.time()
        model(inputs)
        if 'cuda' in str(device):
            torch.cuda.synchronize()
        ed = time.time()
        times.append(ed - st)

    # Convert to [ms] and round to 0.001
    latency_mean = np.round(np.mean(times)*1000, 3)
    latency_std = np.round(np.std(times)*1000, 3)

    return latency_mean, latency_std
