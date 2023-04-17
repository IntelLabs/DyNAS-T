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
from dynast.supernetwork.text_classification.bert_encoding import BertSST2Encoding
from dynast.supernetwork.text_classification.bert_interface import EvaluationInterfaceBertSST2


class BertBase_SST2(SupernetBaseRegisteredClass):
    _name = 'bert_base_sst2'
    _encoding = BertSST2Encoding
    _parameters = {
        'num_layers': {'count': 1, 'vars': [6, 7, 8, 9, 10, 11, 12]},
        'num_attention_heads': {'count': 12, 'vars': [6, 8, 10, 12]},
        'intermediate_size': {'count': 12, 'vars': [1024, 2048, 3072]},
    }
    _evaluation_interface = EvaluationInterfaceBertSST2
    _linas_innerloop_evals = 20000
    _supernet_type = 'text_classification'
    _supernet_metrics = ['latency', 'macs', 'params', 'accuracy_sst2']
