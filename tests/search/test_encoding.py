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
# This software is subject to the terms and conditions entered into between the parties.

import random

import pytest

from dynast.search.encoding import EncodingBase
from dynast.supernetwork.supernetwork_registry import SUPERNET_METRICS, SUPERNET_PARAMETERS


def test_process_param_dict(test_configs):
    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]

        # Test when creating the object (method is called within the init)
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

        assert encoder.mapper == test_config['mapper']
        assert encoder.param_upperbound == test_config['param_upperbound']
        assert encoder.param_count == test_config['param_count']

        # Test method call itself
        assert encoder.process_param_dict() == (
            test_config['mapper'],
            test_config['param_upperbound'],
            test_config['param_count'],
        )


def test_create_inv_mapper(test_configs):
    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]

        # Test when creating the object (method is called within the init)
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)
        assert encoder.inv_mapper == test_config['inv_mapper']

        # Test method call itself
        encoder.mapper = test_config['mapper']
        encoder.create_inv_mapper()
        assert encoder.inv_mapper == test_config['inv_mapper']


def test_random_sample(test_configs):
    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

        assert encoder.random_sample() == test_config['random_samples'][0]


def test_random_samples(test_configs):
    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

        assert encoder.random_samples(size=2) == test_config['random_samples']


def test_import_csv(test_configs):
    supernet_param_dict = SUPERNET_PARAMETERS[test_configs[0]['supernet']]
    encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

    column_names = ['subnet', 'date']
    config_name = column_names[0]
    objective_name = 'macs'

    with pytest.raises(FileNotFoundError):
        encoder.import_csv(
            filepath='i_do_not_exist.csv', config=config_name, objective=objective_name, column_names=column_names
        )

    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

        df = encoder.import_csv(
            filepath=test_config['filepath'],
            config=config_name,
            objective=objective_name,
            column_names=column_names + SUPERNET_METRICS[test_config['supernet']],
            drop_duplicates=False,
        )
        assert len(df) == test_config['dup_entries_in_file']

        df = encoder.import_csv(
            filepath=test_config['filepath'],
            config=config_name,
            objective=objective_name,
            column_names=column_names + SUPERNET_METRICS[test_config['supernet']],
            drop_duplicates=True,
        )
        assert len(df) == test_config['dedup_entries_in_file']


def test_set_seed(test_configs):
    supernet_param_dict = SUPERNET_PARAMETERS[test_configs[0]['supernet']]
    encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

    encoder.set_seed(42)
    assert random.randint(0, 10) == 10

    encoder.set_seed(7)
    assert random.randint(0, 10) == 5


def test_create_training_set(test_configs):
    column_names = ['subnet', 'date']
    config_name = column_names[0]
    objective_name = 'macs'

    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

        df = encoder.import_csv(
            filepath=test_config['filepath'],
            config=config_name,
            objective=objective_name,
            column_names=column_names + SUPERNET_METRICS[test_config['supernet']],
            drop_duplicates=True,
        )

        features, labels = encoder.create_training_set(
            dataframe=df,
            config=config_name,
            train_with_all=True,  #
            seed=42,
        )

        assert all([all([x == y for x, y in zip(a, b)]) for a, b in zip(features, test_config['train_features'])])


def test_encoderbase_onehot_generic(test_configs):
    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

        onehot_vector = encoder.onehot_generic(in_array=test_config['pymoo_vector'])
        assert list(onehot_vector) == test_config['onehot_vector']


def test_encoderbase_translate2param(test_configs):
    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

        supernet_param_dict = encoder.translate2param(test_config['pymoo_vector'])

        assert supernet_param_dict == test_config['param_dict']


def test_encoderbase__translate2pymoo(test_configs):
    for test_config in test_configs:
        supernet_param_dict = SUPERNET_PARAMETERS[test_config['supernet']]
        encoder = EncodingBase(param_dict=supernet_param_dict, verbose=False, seed=42)

        pymoo_vector = encoder.translate2pymoo(test_config['param_dict'])
        assert pymoo_vector == test_config['pymoo_vector']
