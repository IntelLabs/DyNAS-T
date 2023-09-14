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

from dynast.supernetwork.image_classification.ofa_quantization.inc_quantization import (
    _convert_dtype,
    default_policy,
    inc_qconfig_dict,
    qconfig_parse,
    qparam_parse,
)


def test_default_policy():
    policy = default_policy()
    assert type(policy) == dict


def test_qparam_parse():
    parsed = qparam_parse(
        observer_type='minmax',
        bit='float32',
        mode='symmetric',
        granularity='perchannel',
    )

    assert type(parsed) == tuple
    assert len(parsed) == 4
    assert all([type(p) == str for p in parsed])


def test_qparam_parse_invalid_params():
    with pytest.raises(KeyError):
        parsed = qparam_parse(
            observer_type='something_random',
            bit='float32',
            mode='symmetric',
            granularity='perchannel',
        )


def test_qconfig_parse():
    qconfig = qconfig_parse(
        wobserver_type='minmax',
        wbit='float32',
        wmode='symmetric',
        wgranularity='perchannel',
        aobserver_type='kl',
        abit='float32',
        amode='asymmetric',
        agranularity='perchannel',
    )
    assert qconfig == {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}

    qconfig = qconfig_parse(
        wobserver_type='minmax',
        wbit='qint8',
        wmode='symmetric',
        wgranularity='perchannel',
        aobserver_type='kl',
        abit='float32',
        amode='asymmetric',
        agranularity='perchannel',
    )
    assert qconfig == {
        'weight': {'algorithm': ['minmax'], 'dtype': ['int8'], 'scheme': ['sym'], 'granularity': ['per_channel']},
        'activation': {'dtype': ['fp32']},
    }

    qconfig = qconfig_parse(
        wobserver_type='minmax',
        wbit='float32',
        wmode='symmetric',
        wgranularity='perchannel',
        aobserver_type='kl',
        abit='quint8',
        amode='asymmetric',
        agranularity='perchannel',
    )
    assert qconfig == {
        'weight': {'dtype': ['fp32']},
        'activation': {'algorithm': ['kl'], 'dtype': ['uint8'], 'scheme': ['asym'], 'granularity': ['per_channel']},
    }

    qconfig = qconfig_parse(
        wobserver_type='minmax',
        wbit='qint8',
        wmode='symmetric',
        wgranularity='perchannel',
        aobserver_type='kl',
        abit='quint8',
        amode='asymmetric',
        agranularity='perchannel',
    )
    assert qconfig == {
        'weight': {'algorithm': ['minmax'], 'dtype': ['int8'], 'scheme': ['sym'], 'granularity': ['per_channel']},
        'activation': {'algorithm': ['kl'], 'dtype': ['uint8'], 'scheme': ['asym'], 'granularity': ['per_channel']},
    }


def test_convert_dtype_supported_bit_widths():
    assert _convert_dtype(32, 32) == ('float32', 'float32')
    assert _convert_dtype(8, 32) == ('qint8', 'float32')
    assert _convert_dtype(32, 8) == ('float32', 'quint8')
    assert _convert_dtype(8, 8) == ('qint8', 'quint8')


def test_convert_dtype_unsupported_bit_widths_raise_exception():
    with pytest.raises(Exception):
        _convert_dtype(16, 8)
        _convert_dtype(8, 16)


def test_inc_qconfig_dict_modulewise_quantization():
    d = inc_qconfig_dict(
        q_weights_bit=[32, 8, 32, 32],
        q_activations_bit=[32, 8, 32, 32],
        q_weights_mode=['symmetric', 'asymmetric', 'asymmetric', 'asymmetric'],
        q_activations_mode=['asymmetric', 'asymmetric', 'asymmetric', 'asymmetric'],
        q_weights_granularity=['perchannel', 'perchannel', 'perchannel', 'perchannel'],
        q_activations_granularity=['pertensor', 'pertensor', 'pertensor', 'pertensor'],
        regex_module_names=[
            '^(input_stem.0.conv)$',
            '^(input_stem.1.conv.conv)$',
            '^(input_stem.2.conv)$',
            '^(blocks.0.conv1.conv)$',
        ],
    )

    assert d == {
        'model': {'name': 'dynast_quantized_subnet', 'framework': 'pytorch_fx'},
        'device': 'cpu',
        'quantization': {
            'approach': 'post_training_static_quant',
            'calibration': {'sampling_size': 2000},
            'model_wise': {'activation': {'dtype': 'fp32'}, 'weight': {'dtype': 'fp32'}},
            'op_wise': {
                '^(input_stem.0.conv)$': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}},
                '^(input_stem.1.conv.conv)$': {
                    'weight': {
                        'algorithm': ['minmax'],
                        'dtype': ['int8'],
                        'scheme': ['asym'],
                        'granularity': ['per_channel'],
                    },
                    'activation': {
                        'algorithm': ['kl'],
                        'dtype': ['uint8'],
                        'scheme': ['asym'],
                        'granularity': ['per_tensor'],
                    },
                },
                '^(input_stem.2.conv)$': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}},
                '^(blocks.0.conv1.conv)$': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}},
            },
        },
    }


def test_inc_qconfig_dict_global_quantization():
    d = inc_qconfig_dict(
        q_weights_bit=8,
        q_activations_bit=8,
        q_weights_mode='symmetric',
        q_activations_mode='asymmetric',
        q_weights_granularity='perchannel',
        q_activations_granularity='pertensor',
        regex_module_names=None,
    )
    assert d == {
        'model': {'name': 'dynast_quantized_subnet', 'framework': 'pytorch_fx'},
        'device': 'cpu',
        'quantization': {
            'approach': 'post_training_static_quant',
            'calibration': {'sampling_size': 2000},
            'model_wise': {
                'weight': {
                    'algorithm': ['minmax'],
                    'dtype': ['int8'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel'],
                },
                'activation': {
                    'algorithm': ['kl'],
                    'dtype': ['uint8'],
                    'scheme': ['asym'],
                    'granularity': ['per_tensor'],
                },
            },
        },
    }
