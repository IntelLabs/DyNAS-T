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
from dynast.supernetwork.machine_translation.transformer_encoding import TransformerLTEncoding
from dynast.supernetwork.machine_translation.transformer_interface import EvaluationInterfaceTransformerLT


class TransformerLT_WMT_en_de(SupernetBaseRegisteredClass):
    _name = 'transformer_lt_wmt_en_de'
    _encoding = TransformerLTEncoding
    _parameters = {
        'encoder_embed_dim': {'count': 1, 'vars': [640, 512]},
        'decoder_embed_dim': {'count': 1, 'vars': [640, 512]},
        'encoder_ffn_embed_dim': {'count': 6, 'vars': [3072, 2048, 1024]},
        'decoder_ffn_embed_dim': {'count': 6, 'vars': [3072, 2048, 1024]},
        'decoder_layer_num': {'count': 1, 'vars': [6, 5, 4, 3, 2, 1]},
        'encoder_self_attention_heads': {'count': 6, 'vars': [8, 4]},
        'decoder_self_attention_heads': {'count': 6, 'vars': [8, 4]},
        'decoder_ende_attention_heads': {'count': 6, 'vars': [8, 4]},
        'decoder_arbitrary_ende_attn': {'count': 6, 'vars': [-1, 1, 2]},
    }
    _evaluation_interface = EvaluationInterfaceTransformerLT
    _linas_innerloop_evals = 10000
    _supernet_type = 'machine_translation'
    _supernet_metrics = ['latency', 'macs', 'params', 'bleu']
