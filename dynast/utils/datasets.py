# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from dynast.utils import measure_time


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
    def train_dataloader() -> torch.utils.data.DataLoader:
        raise NotImplementedError()

    @staticmethod
    @measure_time
    def validation_dataloader() -> torch.utils.data.DataLoader:
        raise NotImplementedError()

    @staticmethod
    @measure_time
    def test_dataloader() -> torch.utils.data.DataLoader:
        raise NotImplementedError()


class ImageNet(Dataset):
    PATH = "/datasets/imagenet-ilsvrc2012/"

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
        train_dir: str = None,
        shuffle: bool = True,
        num_workers: int = 16,
    ) -> torch.utils.data.DataLoader:
        if not train_dir:
            train_dir = os.path.join(ImageNet.PATH, "train")

        train_dataset = datasets.ImageFolder(
            train_dir,
            ImageNet.train_transforms(image_size),
        )
        train_loader = torch.utils.data.DataLoader(
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
        val_dir: str = None,
        shuffle: bool = False,
        num_workers: int = 16,
    ) -> torch.utils.data.DataLoader:
        if not val_dir:
            val_dir = os.path.join(ImageNet.PATH, "val")

        val_dataset = datasets.ImageFolder(
            val_dir,
            ImageNet.val_transforms(image_size),
        )
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
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
        train_dir: str = None,
        shuffle: bool = True,
        num_workers: int = 16,
    ) -> torch.utils.data.DataLoader:
        if not train_dir:
            train_dir = os.path.join(Imagenette.PATH, "train")

        train_dataset = datasets.ImageFolder(
            train_dir,
            ImageNet.train_transforms(image_size),
        )
        train_loader = torch.utils.data.DataLoader(
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
        val_dir: str = None,
        shuffle: bool = False,
        num_workers: int = 16,
    ) -> torch.utils.data.DataLoader:
        if not val_dir:
            val_dir = os.path.join(Imagenette.PATH, "val")

        val_dataset = datasets.ImageFolder(
            val_dir,
            ImageNet.val_transforms(image_size),
        )
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
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
    ) -> torch.utils.data.DataLoader:
        trainset = torchvision.datasets.CIFAR10(
            root=CIFAR10.PATH,
            train=True,
            download=True,
            transform=CIFAR10.train_transforms(),
        )
        trainloader = torch.utils.data.DataLoader(
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
    ) -> torch.utils.data.DataLoader:
        testset = torchvision.datasets.CIFAR10(
            root=CIFAR10.PATH,
            train=False,
            download=True,
            transform=CIFAR10.val_transforms(),
        )
        testloader = torch.utils.data.DataLoader(
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
    ) -> torch.utils.data.DataLoader:

        image_dataset = datasets.ImageFolder(train_dir, data_transforms)
        return torch.utils.data.DataLoader(
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
    ) -> torch.utils.data.DataLoader:

        image_dataset = datasets.ImageFolder(val_dir, data_transforms)
        return torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
