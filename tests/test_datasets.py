import pytest

from dynast.utils.datasets import CIFAR10, Dataset, ImageNet, Imagenette


def test_dataset_get_by_valid_name():
    valid_dataset_names = [
        ("cifar10", CIFAR10),
        ("CIFAR10", CIFAR10),
        ("cIfAr10", CIFAR10),
        ("imagenet", ImageNet),
        ("imagenette", Imagenette),
    ]

    for ds_name, expected_type in valid_dataset_names:
        obj = Dataset.get(ds_name)
        print(type(obj))
        assert issubclass(obj, Dataset)
        assert isinstance(obj(), expected_type)


def test_dataset_invalid_name_exception():
    invalid_dataset_name = "fancy_dataset"

    with pytest.raises(Exception):
        Dataset.get(invalid_dataset_name)
