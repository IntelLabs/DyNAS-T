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

from dynast.supernetwork import SupernetBaseRegisteredClass
from dynast.supernetwork.image_classification.ofa.ofa_encoding import OFAMobileNetV3Encoding, OFAResNet50Encoding
from dynast.supernetwork.image_classification.ofa.ofa_interface import (
    EvaluationInterfaceOFAMobileNetV3,
    EvaluationInterfaceOFAResNet50,
)


class OFAResNet50Supernet(SupernetBaseRegisteredClass):
    _name = 'ofa_resnet50'
    _encoding = OFAResNet50Encoding
    _parameters = {
        'd': {'count': 5, 'vars': [0, 1, 2]},
        'e': {'count': 18, 'vars': [0.2, 0.25, 0.35]},
        'w': {'count': 6, 'vars': [0, 1, 2]},
    }
    _evaluation_interface = EvaluationInterfaceOFAResNet50
    _linas_innerloop_evals = 5000
    _supernet_type = 'image_classification'
    _supernet_metrics = ['params', 'latency', 'macs', 'accuracy_top1']


class OFAMBv3_d234_e346_k357_w10_Supernet(SupernetBaseRegisteredClass):
    _name = 'ofa_mbv3_d234_e346_k357_w1.0'
    _encoding = OFAMobileNetV3Encoding
    _parameters = {
        'ks': {'count': 20, 'vars': [3, 5, 7]},
        'e': {'count': 20, 'vars': [3, 4, 6]},
        'd': {'count': 5, 'vars': [2, 3, 4]},
    }
    _evaluation_interface = EvaluationInterfaceOFAMobileNetV3
    _linas_innerloop_evals = 20000
    _supernet_type = 'image_classification'
    _supernet_metrics = ['params', 'latency', 'macs', 'accuracy_top1']


class OFAMBv3_d234_e346_k357_w12_Supernet(OFAMBv3_d234_e346_k357_w10_Supernet):
    _name = 'ofa_mbv3_d234_e346_k357_w1.2'


class OFAProxyless_d234_e346_k357_w13_Supernet(OFAMBv3_d234_e346_k357_w10_Supernet):
    _name = 'ofa_proxyless_d234_e346_k357_w1.3'
