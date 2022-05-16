import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from dynast.utils import measure_time


class _Dataset():

    @staticmethod
    @measure_time
    def train_dataloader() -> torch.utils.data.DataLoader:
        raise NotImplementedError()

    @staticmethod
    @measure_time
    def validation_dataloader() -> torch.utils.data.DataLoader:
        raise NotImplementedError()


class ImageNet(_Dataset):

    @staticmethod
    def _transform_normalize() -> transforms.Normalize:
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    @measure_time
    def train_dataloader(
        batch_size: int,
        image_size: int = 224,
        train_dir: str = '/datasets/imagenet-ilsvrc2012/train',
    ) -> torch.utils.data.DataLoader:

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ImageNet._transform_normalize(),
        ])
        train_dataset = datasets.ImageFolder(
            train_dir,
            train_transforms,
        )
        train_sampler = None  # TODO
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),  # TODO
            num_workers=16,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )

        return train_loader

    @staticmethod
    @measure_time
    def validation_dataloader(
        batch_size: int,
        image_size: int = 224,
        val_dir: str = '/datasets/imagenet-ilsvrc2012/val',
    ) -> torch.utils.data.DataLoader:

        val_transforms = transforms.Compose([
            transforms.Resize(int(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            ImageNet._transform_normalize(),
        ])
        val_dataset = datasets.ImageFolder(
            val_dir,
            val_transforms,
        )
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=True,
        )
        return val_loader


class CIFAR10(_Dataset):

    @staticmethod
    def _transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @staticmethod
    @measure_time
    def train_dataloader(
        batch_size: int,
        num_workers=16,
    ) -> torch.utils.data.DataLoader:
        trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                                download=True, transform=CIFAR10._transform())  # TODO Param download
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
        return trainloader

    @staticmethod
    @measure_time
    def validation_dataloader(
        batch_size: int,
        num_workers=16,
    ) -> torch.utils.data.DataLoader:
        testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                               download=True, transform=CIFAR10._transform())
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        return testloader
