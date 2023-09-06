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


from dynast.utils.nn import AverageMeter, get_macs, get_parameters, measure_latency, validate_classification
from tests.helpers import TinyModel, get_random_dataloader


def test_average_meter():
    am = AverageMeter()

    val = 1
    am.update(val=val)
    assert am.val == val
    assert am.sum == val
    assert am.count == 1
    assert am.avg == 1

    val = 2
    am.update(val=val, n=2)
    assert am.val == val
    assert am.sum == 5
    assert am.count == 3
    assert am.avg == 5 / 3

    am.reset()

    assert am.val == 0
    assert am.sum == 0
    assert am.count == 0
    assert am.avg == 0


def test_measure_latency(device):
    model = TinyModel()

    lat_avg, lat_std = measure_latency(
        model=model,
        input_size=(1, 1, 32, 32),
        device=device,
        warmup_steps=1,
        measure_steps=1,
    )


def test_get_macs(device):
    model = TinyModel()

    macs = get_macs(
        model=model,
        input_size=(1, 1, 32, 32),
        device=device,
    )

    assert model.ground_truth_macs == macs


def test_get_parameters(device):
    model = TinyModel()

    params = get_parameters(
        model=model,
        device=device,
    )

    assert model.ground_truth_params == params


def test_validate_classification(device):
    model = TinyModel()

    result = validate_classification(
        model=model,
        data_loader=get_random_dataloader(),
        device=device,
    )

    assert len(result) == 3
