import argparse
import logging
from typing import Tuple

import torchvision
import torchvision.models as models
from torch import nn

from dynast.utils import log, measure_time, set_logger
from dynast.utils.datasets import Dataset, ImageNet
from dynast.utils.nn import measure_latency, validate_classification


def get_torchvision_model(
    model_name: str,
) -> nn.Module:
    try:
        model = getattr(models, model_name)(pretrained=True)
        model.eval()
        return model
    except AttributeError as ae:
        log.error(
            'Model {model_name} not available. This can be due to either a typo or the model is not '
            'available in torchvision=={torchvision_version}. \nAvailable models: {available_models}'.format(
                model_name=model_name,
                torchvision_version=torchvision.__version__,
                available_models=', '.join([m for m in dir(models) if not m.startswith('_')]),
            )
        )
        raise ae


class Reference(object):
    @measure_time
    def validate(
        self,
        device: str = 'cpu',
        batch_size: int = 128,
        input_size: int = 224,
        test_size: int = None,
    ):
        raise NotImplementedError()

    @measure_time
    def benchmark(
        self,
        device: str = 'cpu',
        batch_size: int = 128,
        input_size: int = 224,
        warmup_steps: int = 10,
        measure_steps: int = 50,
    ):
        raise NotImplementedError()


class TorchVisionReference(Reference):
    def __init__(
        self,
        model_name: str,
        dataset: Dataset = ImageNet,
    ) -> None:
        self.model_name = model_name
        self.dataset = dataset

        log.info('{name} for \'{model_name}\' on \'{dataset_name}\' dataset'.format(
            name=str(self),
            model_name=self.model_name,
            dataset_name=self.dataset.name(),
        ))
        self.model = get_torchvision_model(model_name=self.model_name)

    @measure_time
    def validate(
        self,
        device: str = 'cpu',
        batch_size: int = 128,
        input_size: int = 224,
        test_size: int = None,
    ) -> Tuple[float, float, float]:
        model = self.model.to(device)
        loss, top1, top5 = validate_classification(
            model=model,
            device=device,
            is_openvino=False,
            batch_size=batch_size,
            data_loader=self.dataset.validation_dataloader(
                batch_size=batch_size,
                image_size=input_size,
            ),
            test_size=test_size,
        )
        log.info('\'{model_name}\' on \'{dataset_name}\' - top1 {top1} top5 {top5} loss {loss}'.format(
            model_name=self.model_name,
            dataset_name=self.dataset.name(),
            top1=top1,
            top5=top5,
            loss=loss,
        ))
        return loss, top1, top5

    @measure_time
    def benchmark(
        self,
        device: str = 'cpu',
        batch_size: int = 128,
        input_size: int = 224,
        warmup_steps: int = 10,
        measure_steps: int = 50,
    ) -> Tuple[float, float]:
        model = self.model.to(device)
        latency_mean, latency_std = measure_latency(
            model=model,
            input_size=(batch_size, 3, input_size, input_size),
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            device=device,
        )
        log.info('\'{model_name}\' (BS={batch_size}) mean latency {latency_mean} +/- {latency_std}'.format(
            model_name=self.model_name,
            batch_size=batch_size,
            latency_mean=latency_mean,
            latency_std=latency_std,
        ))
        return latency_mean, latency_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-t', '--test_size', type=int, default=None,
                        help='How many batches should be used for validation.')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='How many batches should be used to warm up latency measurement when benchmarking.')
    parser.add_argument('--measure_steps', type=int, default=50,
                        help='How many batches should be used for actual latency measurement when benchmarking.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'imagenette', 'cifar10'], default='imagenet')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args()

    if args.debug:
        set_logger(logging.DEBUG)

    log.info('Settings: {}'.format(args))

    ref = TorchVisionReference(
        model_name=args.model,
        dataset=Dataset.get(args.dataset),
    )

    ref.validate(
        device=args.device,
        batch_size=args.batch_size,
        test_size=args.test_size,
    )
    ref.benchmark(
        device=args.device,
        batch_size=args.batch_size,
        input_size=args.input_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )
