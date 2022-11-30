# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

from dynast.utils.datasets import Dataset
from dynast.utils.reference import TorchVisionReference


def test_reference_torch_accuracy(device, dataset_name, batch_size):
    ref = TorchVisionReference(
        model_name="resnet50",
        dataset=Dataset.get(dataset_name),
        quantize=False,
    )

    # Note: Accuracy will be relatively high due to small test size.
    _, top1, _ = ref.validate(
        device=device,
        batch_size=batch_size,
        test_size=10,
    )

    expected_top1_range = (88.0, 89.0)  # 76.156 for validation with full ImageNet

    # There might be slight variations in accuracy, depending on the backend, torch version etc.,
    # so we check if the result is within 1% error.
    assert min(expected_top1_range) <= top1 <= max(expected_top1_range)


def test_reference_torch_quantized_accuracy(device, dataset_name, batch_size):
    # NOTE(macsz) CPU only. Crashes on GPU (tested on torch 1.11).

    ref = TorchVisionReference(
        model_name="resnet50",
        dataset=Dataset.get(dataset_name),
        quantize=True,
    )

    # Note: Accuracy will be relatively high due to small test size.
    _, top1, _ = ref.validate(
        device=device,
        batch_size=batch_size,
        test_size=10,
    )

    expected_top1_range = (
        88.0,
        89.0,
    )  # 75.942 for validation with full ImageNet, on small batch it's the same as FP32 model

    # There might be slight variations in accuracy, depending on the backend, torch version etc.,
    # so we check if the result is within 1% error.
    assert min(expected_top1_range) <= top1 <= max(expected_top1_range)


def test_reference_torch_gflops(device, dataset_name, batch_size):
    ref = TorchVisionReference(
        model_name="resnet50",
        dataset=Dataset.get(dataset_name),
        quantize=False,
    )

    assert ref.get_gflops() == 4.111512576


def test_reference_torch_params(device, dataset_name, batch_size):
    ref = TorchVisionReference(
        model_name="resnet50",
        dataset=Dataset.get(dataset_name),
        quantize=False,
    )

    assert ref.get_params() == 25557032
