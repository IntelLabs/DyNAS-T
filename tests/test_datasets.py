# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

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
