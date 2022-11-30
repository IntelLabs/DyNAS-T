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
import torch.nn as nn
import torch.nn.functional as F

from dynast.utils.onnx import is_onnx, load_onnx_model, save_onnx_model


def swish(x):
    return x * F.sigmoid(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_convert_to_onnx_and_check_type():
    # TODO(Maciej) This is probably a golden example on how not to write UTs.
    # Use fixtures in the futurue.
    simple_dnn = SimpleCNN()
    p = "/tmp/model.onnx"
    try:
        os.remove(p)
    except FileNotFoundError:
        pass
    save_onnx_model(network=simple_dnn, model_path=p, input_shape=(1, 3, 32, 32))
    onnx_model = load_onnx_model(p)
    assert is_onnx(onnx_model)
    os.remove(p)


def test_is_onnx_false():
    simple_dnn = SimpleCNN()
    assert not is_onnx(simple_dnn)
