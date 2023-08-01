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


from collections import OrderedDict

import pytest

from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_encoding import BootstrapNASEncoding
from dynast.utils import LazyImport

elasticity_dim = LazyImport('nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim')


class TestBootstrapNASEncoding:
    # fmt: off
    bootstrapnas_supernet_parameters = {
        'width': {
            0: [256, 200, 152, 128], 1: [512, 408, 304, 256], 2: [1024, 816, 608, 512], 3: [2048, 1632, 1224, 1024],
            4: [64, 48, 32], 5: [64, 48, 32], 6: [64, 48, 32], 7: [64, 48, 32], 8: [64, 48, 32], 9: [64, 48, 32],
            10: [64, 48, 32], 11: [128, 96, 72, 64], 12: [128, 96, 72, 64], 13: [128, 96, 72, 64],
            14: [128, 96, 72, 64], 15: [128, 96, 72, 64], 16: [128, 96, 72, 64], 17: [128, 96, 72, 64],
            18: [128, 96, 72, 64], 19: [256, 200, 152, 128], 20: [256, 200, 152, 128], 21: [256, 200, 152, 128],
            22: [256, 200, 152, 128], 23: [256, 200, 152, 128], 24: [256, 200, 152, 128], 25: [256, 200, 152, 128],
            26: [256, 200, 152, 128], 27: [256, 200, 152, 128], 28: [256, 200, 152, 128], 29: [256, 200, 152, 128],
            30: [256, 200, 152, 128], 31: [512, 408, 304, 256], 32: [512, 408, 304, 256], 33: [512, 408, 304, 256],
            34: [512, 408, 304, 256], 35: [512, 408, 304, 256], 36: [512, 408, 304, 256]
        },
        'depth': [
            [0], [2], [4], [5], [0, 2], [0, 4], [0, 5], [1, 2], [2, 4], [2, 5], [3, 4], [4, 5], [0, 1, 2], [0, 2, 4],
            [0, 2, 5], [0, 3, 4], [0, 4, 5], [1, 2, 4], [1, 2, 5], [2, 3, 4], [2, 4, 5], [3, 4, 5], [0, 1, 2, 4],
            [0, 1, 2, 5], [0, 2, 3, 4], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 4, 5], [2, 3, 4, 5],
            [0, 1, 2, 3, 4], [0, 1, 2, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], []
        ]
    }
    dynast_supernet_parameters = {
        'width_0': {'count': 1, 'vars': [256, 200, 152, 128]}, 'width_1': {'count': 1, 'vars': [512, 408, 304, 256]},
        'width_2': {'count': 1, 'vars': [1024, 816, 608, 512]},
        'width_3': {'count': 1, 'vars': [2048, 1632, 1224, 1024]}, 'width_4': {'count': 1, 'vars': [64, 48, 32]},
        'width_5': {'count': 1, 'vars': [64, 48, 32]}, 'width_6': {'count': 1, 'vars': [64, 48, 32]},
        'width_7': {'count': 1, 'vars': [64, 48, 32]}, 'width_8': {'count': 1, 'vars': [64, 48, 32]},
        'width_9': {'count': 1, 'vars': [64, 48, 32]}, 'width_10': {'count': 1, 'vars': [64, 48, 32]},
        'width_11': {'count': 1, 'vars': [128, 96, 72, 64]}, 'width_12': {'count': 1, 'vars': [128, 96, 72, 64]},
        'width_13': {'count': 1, 'vars': [128, 96, 72, 64]}, 'width_14': {'count': 1, 'vars': [128, 96, 72, 64]},
        'width_15': {'count': 1, 'vars': [128, 96, 72, 64]}, 'width_16': {'count': 1, 'vars': [128, 96, 72, 64]},
        'width_17': {'count': 1, 'vars': [128, 96, 72, 64]}, 'width_18': {'count': 1, 'vars': [128, 96, 72, 64]},
        'width_19': {'count': 1, 'vars': [256, 200, 152, 128]}, 'width_20': {'count': 1, 'vars': [256, 200, 152, 128]},
        'width_21': {'count': 1, 'vars': [256, 200, 152, 128]}, 'width_22': {'count': 1, 'vars': [256, 200, 152, 128]},
        'width_23': {'count': 1, 'vars': [256, 200, 152, 128]}, 'width_24': {'count': 1, 'vars': [256, 200, 152, 128]},
        'width_25': {'count': 1, 'vars': [256, 200, 152, 128]}, 'width_26': {'count': 1, 'vars': [256, 200, 152, 128]},
        'width_27': {'count': 1, 'vars': [256, 200, 152, 128]}, 'width_28': {'count': 1, 'vars': [256, 200, 152, 128]},
        'width_29': {'count': 1, 'vars': [256, 200, 152, 128]}, 'width_30': {'count': 1, 'vars': [256, 200, 152, 128]},
        'width_31': {'count': 1, 'vars': [512, 408, 304, 256]}, 'width_32': {'count': 1, 'vars': [512, 408, 304, 256]},
        'width_33': {'count': 1, 'vars': [512, 408, 304, 256]}, 'width_34': {'count': 1, 'vars': [512, 408, 304, 256]},
        'width_35': {'count': 1, 'vars': [512, 408, 304, 256]}, 'width_36': {'count': 1, 'vars': [512, 408, 304, 256]},
        'depth': {'count': 1, 'vars': [
            [0], [2], [4], [5], [0, 2], [0, 4], [0, 5], [1, 2], [2, 4], [2, 5], [3, 4], [4, 5], [0, 1, 2], [0, 2, 4],
            [0, 2, 5], [0, 3, 4], [0, 4, 5], [1, 2, 4], [1, 2, 5], [2, 3, 4], [2, 4, 5], [3, 4, 5], [0, 1, 2, 4],
            [0, 1, 2, 5], [0, 2, 3, 4], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 4, 5], [2, 3, 4, 5],
            [0, 1, 2, 3, 4], [0, 1, 2, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], []]
        }
    }
    parameter_mapper=[
        {0: 256, 1: 200, 2: 152, 3: 128}, {0: 512, 1: 408, 2: 304, 3: 256}, {0: 1024, 1: 816, 2: 608, 3: 512},
        {0: 2048, 1: 1632, 2: 1224, 3: 1024}, {0: 64, 1: 48, 2: 32}, {0: 64, 1: 48, 2: 32}, {0: 64, 1: 48, 2: 32},
        {0: 64, 1: 48, 2: 32}, {0: 64, 1: 48, 2: 32}, {0: 64, 1: 48, 2: 32}, {0: 64, 1: 48, 2: 32},
        {0: 128, 1: 96, 2: 72, 3: 64}, {0: 128, 1: 96, 2: 72, 3: 64}, {0: 128, 1: 96, 2: 72, 3: 64},
        {0: 128, 1: 96, 2: 72, 3: 64}, {0: 128, 1: 96, 2: 72, 3: 64}, {0: 128, 1: 96, 2: 72, 3: 64},
        {0: 128, 1: 96, 2: 72, 3: 64}, {0: 128, 1: 96, 2: 72, 3: 64}, {0: 256, 1: 200, 2: 152, 3: 128},
        {0: 256, 1: 200, 2: 152, 3: 128}, {0: 256, 1: 200, 2: 152, 3: 128}, {0: 256, 1: 200, 2: 152, 3: 128},
        {0: 256, 1: 200, 2: 152, 3: 128}, {0: 256, 1: 200, 2: 152, 3: 128}, {0: 256, 1: 200, 2: 152, 3: 128},
        {0: 256, 1: 200, 2: 152, 3: 128}, {0: 256, 1: 200, 2: 152, 3: 128}, {0: 256, 1: 200, 2: 152, 3: 128},
        {0: 256, 1: 200, 2: 152, 3: 128}, {0: 256, 1: 200, 2: 152, 3: 128}, {0: 512, 1: 408, 2: 304, 3: 256},
        {0: 512, 1: 408, 2: 304, 3: 256}, {0: 512, 1: 408, 2: 304, 3: 256}, {0: 512, 1: 408, 2: 304, 3: 256},
        {0: 512, 1: 408, 2: 304, 3: 256}, {0: 512, 1: 408, 2: 304, 3: 256},
        {
            0: '[0]', 1: '[2]', 2: '[4]', 3: '[5]', 4: '[0, 2]', 5: '[0, 4]', 6: '[0, 5]', 7: '[1, 2]', 8: '[2, 4]',
            9: '[2, 5]', 10: '[3, 4]', 11: '[4, 5]', 12: '[0, 1, 2]', 13: '[0, 2, 4]', 14: '[0, 2, 5]',
            15: '[0, 3, 4]', 16: '[0, 4, 5]', 17: '[1, 2, 4]', 18: '[1, 2, 5]', 19: '[2, 3, 4]', 20: '[2, 4, 5]',
            21: '[3, 4, 5]', 22: '[0, 1, 2, 4]', 23: '[0, 1, 2, 5]', 24: '[0, 2, 3, 4]', 25: '[0, 2, 4, 5]',
            26: '[0, 3, 4, 5]', 27: '[1, 2, 3, 4]', 28: '[1, 2, 4, 5]', 29: '[2, 3, 4, 5]', 30: '[0, 1, 2, 3, 4]',
            31: '[0, 1, 2, 4, 5]', 32: '[0, 2, 3, 4, 5]', 33: '[1, 2, 3, 4, 5]', 34: '[0, 1, 2, 3, 4, 5]', 35: '[]'
        }
    ]
    param_upperbound=[
        3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 35
    ]
    param_count=38
    pymoo_vector=[
        0, 0, 2, 1, 0, 0, 2, 0, 2, 2, 2, 0, 3, 0, 0, 0, 1, 1, 0,
        1, 3, 1, 3, 2, 0, 1, 3, 2, 2, 1, 1, 2, 0, 0, 3, 0, 2, 22
    ]
    param_dict = {
        'width_0': [256], 'width_1': [512], 'width_2': [608], 'width_3': [1632], 'width_4': [64], 'width_5': [64],
        'width_6': [32], 'width_7': [64], 'width_8': [32], 'width_9': [32], 'width_10': [32], 'width_11': [128],
        'width_12': [64], 'width_13': [128], 'width_14': [128], 'width_15': [128], 'width_16': [96], 'width_17': [96],
        'width_18': [128], 'width_19': [200], 'width_20': [128], 'width_21': [200], 'width_22': [128],
        'width_23': [152], 'width_24': [256], 'width_25': [200], 'width_26': [128], 'width_27': [152],
        'width_28': [152], 'width_29': [200], 'width_30': [200], 'width_31': [304], 'width_32': [512],
        'width_33': [512], 'width_34': [256], 'width_35': [512], 'width_36': [304], 'depth': ['[0, 1, 2, 4]']
    }
    # fmt: on

    def test_bnas_to_dynast(self):
        bootstrapnas_encoding = BootstrapNASEncoding(self.bootstrapnas_supernet_parameters)
        assert self.dynast_supernet_parameters == bootstrapnas_encoding.param_dict

        # fmt: off
        invalid_bootstrapnas_supernet_parameters = {'unknown_key': {0: [256, 200, 152, 128], 1: [512, 408, 304, 256], 2: [1024, 816, 608, 512], 3: [2048, 1632, 1224, 1024], 4: [64, 48, 32], 5: [64, 48, 32], 6: [64, 48, 32], 7: [64, 48, 32], 8: [64, 48, 32], 9: [64, 48, 32], 10: [64, 48, 32], 11: [128, 96, 72, 64], 12: [128, 96, 72, 64], 13: [128, 96, 72, 64], 14: [128, 96, 72, 64], 15: [128, 96, 72, 64], 16: [128, 96, 72, 64], 17: [128, 96, 72, 64], 18: [128, 96, 72, 64], 19: [256, 200, 152, 128], 20: [256, 200, 152, 128], 21: [256, 200, 152, 128], 22: [256, 200, 152, 128], 23: [256, 200, 152, 128], 24: [256, 200, 152, 128], 25: [256, 200, 152, 128], 26: [256, 200, 152, 128], 27: [256, 200, 152, 128], 28: [256, 200, 152, 128], 29: [256, 200, 152, 128], 30: [256, 200, 152, 128], 31: [512, 408, 304, 256], 32: [512, 408, 304, 256], 33: [512, 408, 304, 256], 34: [512, 408, 304, 256], 35: [512, 408, 304, 256], 36: [512, 408, 304, 256]}, 'depth': [[0], [2], [4], [5], [0, 2], [0, 4], [0, 5], [1, 2], [2, 4], [2, 5], [3, 4], [4, 5], [0, 1, 2], [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 4, 5], [1, 2, 4], [1, 2, 5], [2, 3, 4], [2, 4, 5], [3, 4, 5], [0, 1, 2, 4], [0, 1, 2, 5], [0, 2, 3, 4], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 4, 5], [2, 3, 4, 5], [0, 1, 2, 3, 4], [0, 1, 2, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], []]}
        # fmt: on
        with pytest.raises(KeyError):
            bootstrapnas_encoding = BootstrapNASEncoding(invalid_bootstrapnas_supernet_parameters)

    def test_process_param_dict(self):
        bootstrapnas_encoding = BootstrapNASEncoding(self.bootstrapnas_supernet_parameters)
        parameter_mapper, param_upperbound, param_count = (
            bootstrapnas_encoding.mapper,
            bootstrapnas_encoding.param_upperbound,
            bootstrapnas_encoding.param_count,
        )

        assert parameter_mapper == self.parameter_mapper
        assert param_upperbound == self.param_upperbound
        assert param_count == self.param_count

    def test_translate2param(self):
        bootstrapnas_encoding = BootstrapNASEncoding(self.bootstrapnas_supernet_parameters)
        assert bootstrapnas_encoding.translate2param(self.pymoo_vector) == self.param_dict

    def test_convert_subnet_config_to_bootstrapnas(self):
        dynast_subnet_config = {
            'width_0': [256],
            'width_1': [512],
            'width_2': [608],
            'width_3': [1632],
            'width_4': [64],
            'width_5': [64],
            'depth': ['[0, 1, 2, 4]'],
        }
        bootstrapnas_subnet_config = BootstrapNASEncoding.convert_subnet_config_to_bootstrapnas(
            subnet_config=dynast_subnet_config,
        )
        expected_bootstrapnas_subnet_config = OrderedDict(
            [
                (elasticity_dim.ElasticityDim.WIDTH, {0: 256, 1: 512, 2: 608, 3: 1632, 4: 64, 5: 64}),
                (elasticity_dim.ElasticityDim.DEPTH, [0, 1, 2, 4]),
            ]
        )
        assert expected_bootstrapnas_subnet_config == bootstrapnas_subnet_config

        dynast_subnet_config_str = "{'width_0': [256], 'width_1': [512], 'width_2': [608], 'width_3': [1632], 'width_4': [64], 'width_5': [64], 'depth': ['[0, 1, 2, 4]']}"
        bootstrapnas_subnet_config = BootstrapNASEncoding.convert_subnet_config_to_bootstrapnas(
            subnet_config=dynast_subnet_config_str,
        )
        assert expected_bootstrapnas_subnet_config == bootstrapnas_subnet_config
