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

import pytest

from dynast import DyNAS
from dynast.search.search_tactic import LINAS, Evolutionary, LINASDistributed, RandomSearch, RandomSearchDistributed


def test_dynas_optimization_metrics_unsupported_number():
    with pytest.raises(SystemExit):
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3', 'm4'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='linas',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        )


def test_dynas_unsupported_search_tactic_exception():
    with pytest.raises(NotImplementedError):
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='new_and_unsupported_search_tactic',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        )


def test_dynas_supported_search_tactics():
    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['latency', 'macs'],
            measurements=['latency', 'macs'],
            search_tactic='linas',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        LINAS,
    )

    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['latency', 'macs'],
            measurements=['latency', 'macs'],
            search_tactic='evolutionary',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        Evolutionary,
    )

    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['latency', 'macs'],
            measurements=['latency', 'macs'],
            search_tactic='random',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        RandomSearch,
    )

    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['latency', 'macs'],
            measurements=['latency', 'macs'],
            search_tactic='linas',
            distributed=True,
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        LINASDistributed,
    )

    assert isinstance(
        DyNAS(
            supernet='ofa_resnet50',
            optimization_metrics=['latency', 'macs'],
            measurements=['latency', 'macs'],
            search_tactic='random',
            distributed=True,
            num_evals=1,
            results_path='test',
            dataset_path='test',
        ),
        RandomSearchDistributed,
    )


def test_dynas_unsupported_supernet():
    with pytest.raises(Exception):
        DyNAS(
            supernet='unsupported_supernet',
            optimization_metrics=['m1', 'm2', 'm3'],
            measurements=['m1', 'm2', 'm3'],
            search_tactic='linas',
            num_evals=1,
            results_path='test',
            dataset_path='test',
        )


def test_dynas_eval_batch_size():
    input_kwargs = {
        'supernet': 'ofa_resnet50',
        'batch_size': 32,
    }
    expected_kwargs = {
        'supernet': 'ofa_resnet50',
        'batch_size': 32,
        'eval_batch_size': 32,
    }
    DyNAS._set_eval_batch_size(input_kwargs)

    assert input_kwargs == expected_kwargs

    input_kwargs = {
        'supernet': 'ofa_resnet50',
        'batch_size': 32,
        'eval_batch_size': 64,
    }
    expected_kwargs = {
        'supernet': 'ofa_resnet50',
        'batch_size': 32,
        'eval_batch_size': 64,
    }
    DyNAS._set_eval_batch_size(input_kwargs)

    assert input_kwargs == expected_kwargs
