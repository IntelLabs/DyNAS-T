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

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim

from dynast.supernetwork.image_classification.bootstrapnas.bootstrapnas_encoding import BootstrapNASEncoding


def test_convert_subnet_config_to_bootstrapnas():
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
            (ElasticityDim.WIDTH, {0: 256, 1: 512, 2: 608, 3: 1632, 4: 64, 5: 64}),
            (ElasticityDim.DEPTH, [0, 1, 2, 4]),
        ]
    )
    assert expected_bootstrapnas_subnet_config == bootstrapnas_subnet_config
