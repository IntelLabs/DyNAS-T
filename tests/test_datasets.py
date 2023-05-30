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
        assert issubclass(obj, Dataset)
        assert isinstance(obj(), expected_type)


def test_dataset_invalid_name_exception():
    invalid_dataset_name = "fancy_dataset"

    with pytest.raises(Exception):
        Dataset.get(invalid_dataset_name)


@pytest.mark.skipif(
    os.path.exists(not "/datasets/imagenet-ilsvrc2012/val"),
    reason="ImageNet dataset not found; skipping test (TODO(macsz) should be mocked in the future!)",
)
def test_dataset_test_fraction():
    ImageNet.PATH = "/datasets/imagenet-ilsvrc2012/"
    dataset = ImageNet()
    bs = 128
    imagenet_val_steps = 50000 // bs
    assert len(dataset.validation_dataloader(bs, fraction=1.0)) == imagenet_val_steps
    assert len(dataset.validation_dataloader(bs, fraction=0.0)) == imagenet_val_steps
    assert len(dataset.validation_dataloader(bs, fraction=0.2)) == imagenet_val_steps // 5
