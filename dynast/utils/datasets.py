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

import os
from typing import Optional

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dynast.utils import measure_time


def _dataset_fraction(dataset: torchvision.datasets.DatasetFolder, fraction: float, seed: int = 21):
    # Use random subset of validation data if valid fraction specified
    if (fraction > 0.0) and (fraction < 1.0):
        torch.manual_seed(seed)
        random_indices = torch.randperm(len(dataset))
        example_count = round(fraction * len(dataset))
        dataset = torch.utils.data.Subset(dataset, random_indices[:example_count])
    return dataset


class Dataset(object):
    PATH = ""

    @staticmethod
    def name():
        raise NotImplementedError()

    @staticmethod
    def get(dataset_name: str) -> "Dataset":
        dataset_name = dataset_name.lower()
        if "imagenet" == dataset_name:
            return ImageNet
        elif "imagenette" == dataset_name:
            return Imagenette
        elif "cifar10" == dataset_name:
            return CIFAR10
        else:
            raise Exception("Not a valid dataset name.")

    @staticmethod
    def train_transforms() -> transforms.Compose:
        raise NotImplementedError()

    @staticmethod
    def val_transforms() -> transforms.Compose:
        raise NotImplementedError()

    @staticmethod
    @measure_time
    def train_dataloader() -> DataLoader:
        raise NotImplementedError()

    @staticmethod
    @measure_time
    def validation_dataloader() -> DataLoader:
        raise NotImplementedError()

    @staticmethod
    @measure_time
    def test_dataloader() -> DataLoader:
        raise NotImplementedError()


class ImageNet(Dataset):
    PATH: str = "/datasets/imagenet-ilsvrc2012/"

    @staticmethod
    def name() -> str:
        return "ImageNet"

    @staticmethod
    def _transform_normalize() -> transforms.Normalize:
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def train_transforms(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ImageNet._transform_normalize(),
            ]
        )

    @staticmethod
    def val_transforms(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size / 0.875)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                ImageNet._transform_normalize(),
            ]
        )

    @staticmethod
    @measure_time
    def train_dataloader(
        batch_size: int,
        image_size: int = 224,
        train_dir: Optional[str] = None,
        shuffle: bool = True,
        num_workers: int = 16,
    ) -> DataLoader:
        if not train_dir:
            train_dir = os.path.join(ImageNet.PATH, "train")

        train_dataset = datasets.ImageFolder(
            train_dir,
            ImageNet.train_transforms(image_size),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,
        )

        return train_loader

    @staticmethod
    @measure_time
    def validation_dataloader(
        batch_size: int,
        image_size: int = 224,
        val_dir: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 16,
        fraction: float = 1.0,
    ) -> DataLoader:
        if not val_dir:
            val_dir = os.path.join(ImageNet.PATH, "val")

        val_dataset = datasets.ImageFolder(
            val_dir,
            ImageNet.val_transforms(image_size),
        )

        val_dataset = _dataset_fraction(val_dataset, fraction)

        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            sampler=val_sampler,
            drop_last=True,
        )
        return val_loader


class Imagenette(Dataset):
    """Imagenette is a subset of 10 easily classified classes from Imagenet.
    More: https://github.com/fastai/imagenette
    """

    PATH = "/store/nosnap/datasets/imagenette2-320"

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "Imagenette"

    @staticmethod
    @measure_time
    def train_dataloader(
        batch_size: int,
        image_size: int = 224,
        train_dir: Optional[str] = None,
        shuffle: bool = True,
        num_workers: int = 16,
    ) -> DataLoader:
        if not train_dir:
            train_dir = os.path.join(Imagenette.PATH, "train")

        train_dataset = datasets.ImageFolder(
            train_dir,
            ImageNet.train_transforms(image_size),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,
        )

        return train_loader

    @staticmethod
    @measure_time
    def validation_dataloader(
        batch_size: int,
        image_size: int = 224,
        val_dir: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 16,
        fraction: float = 1.0,
    ) -> DataLoader:
        if not val_dir:
            val_dir = os.path.join(Imagenette.PATH, "val")

        val_dataset = datasets.ImageFolder(
            val_dir,
            ImageNet.val_transforms(image_size),
        )

        val_dataset = _dataset_fraction(val_dataset, fraction)

        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            sampler=val_sampler,
            drop_last=True,
        )
        return val_loader


class CIFAR10(Dataset):
    PATH = "./cifar10"

    @staticmethod
    def name() -> str:
        return "CIFAR10"

    @staticmethod
    def _transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def train_transforms() -> transforms.Compose:
        return CIFAR10._transform()

    @staticmethod
    def val_transforms() -> transforms.Compose:
        return CIFAR10._transform()

    @staticmethod
    @measure_time
    def train_dataloader(
        batch_size: int,
        shuffle: bool = True,
        num_workers=16,
    ) -> DataLoader:
        trainset = torchvision.datasets.CIFAR10(
            root=CIFAR10.PATH,
            train=True,
            download=True,
            transform=CIFAR10.train_transforms(),
        )
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        return trainloader

    @staticmethod
    @measure_time
    def validation_dataloader(
        batch_size: int,
        shuffle: bool = False,
        num_workers=16,
        fraction: float = 1.0,
    ) -> DataLoader:
        testset = torchvision.datasets.CIFAR10(
            root=CIFAR10.PATH,
            train=False,
            download=True,
            transform=CIFAR10.val_transforms(),
        )

        testset = _dataset_fraction(testset, fraction)

        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        return testloader


class CustomDataset(Dataset):
    @staticmethod
    def name() -> str:
        return "CustomDataset"

    @staticmethod
    @measure_time
    def train_dataloader(
        train_dir: str,
        batch_size: int,
        data_transforms: transforms.Compose,
        shuffle: bool = True,
        num_workers: int = 16,
    ) -> DataLoader:
        image_dataset = datasets.ImageFolder(train_dir, data_transforms)
        return DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    @staticmethod
    @measure_time
    def validation_dataloader(
        val_dir: str,
        batch_size: int,
        data_transforms: transforms.Compose,
        shuffle: bool = False,
        num_workers: int = 16,
        fraction: float = 1.0,
    ) -> DataLoader:
        image_dataset = datasets.ImageFolder(val_dir, data_transforms)

        image_dataset = _dataset_fraction(image_dataset, fraction)

        return DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
