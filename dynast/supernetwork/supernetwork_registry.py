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


from dynast.supernetwork.image_classification.ofa.ofa_encoding import OFAMobileNetV3Encoding, OFAResNet50Encoding
from dynast.supernetwork.image_classification.ofa.ofa_interface import (
    EvaluationInterfaceOFAMobileNetV3,
    EvaluationInterfaceOFAResNet50,
)
from dynast.supernetwork.machine_translation.transformer_encoding import TransformerLTEncoding
from dynast.supernetwork.machine_translation.transformer_interface import EvaluationInterfaceTransformerLT
from dynast.supernetwork.text_classification.bert_encoding import BertSST2Encoding
from dynast.supernetwork.text_classification.bert_interface import EvaluationInterfaceBertSST2

SUPERNET_ENCODING = {
    'ofa_resnet50': OFAResNet50Encoding,
    'ofa_mbv3_d234_e346_k357_w1.0': OFAMobileNetV3Encoding,
    'ofa_mbv3_d234_e346_k357_w1.2': OFAMobileNetV3Encoding,
    'ofa_proxyless_d234_e346_k357_w1.3': OFAMobileNetV3Encoding,
    'transformer_lt_wmt_en_de': TransformerLTEncoding,
    'bert_base_sst2': BertSST2Encoding,
}

SUPERNET_PARAMETERS = {
    'ofa_resnet50': {
        'd': {'count': 5, 'vars': [0, 1, 2]},
        'e': {'count': 18, 'vars': [0.2, 0.25, 0.35]},
        'w': {'count': 6, 'vars': [0, 1, 2]},
    },
    'ofa_mbv3_d234_e346_k357_w1.0': {
        'ks': {'count': 20, 'vars': [3, 5, 7]},
        'e': {'count': 20, 'vars': [3, 4, 6]},
        'd': {'count': 5, 'vars': [2, 3, 4]},
    },
    'ofa_mbv3_d234_e346_k357_w1.2': {
        'ks': {'count': 20, 'vars': [3, 5, 7]},
        'e': {'count': 20, 'vars': [3, 4, 6]},
        'd': {'count': 5, 'vars': [2, 3, 4]},
    },
    'ofa_proxyless_d234_e346_k357_w1.3': {
        'ks': {'count': 20, 'vars': [3, 5, 7]},
        'e': {'count': 20, 'vars': [3, 4, 6]},
        'd': {'count': 5, 'vars': [2, 3, 4]},
    },
    'transformer_lt_wmt_en_de': {
        'encoder_embed_dim': {'count': 1, 'vars': [640, 512]},
        'decoder_embed_dim': {'count': 1, 'vars': [640, 512]},
        'encoder_ffn_embed_dim': {'count': 6, 'vars': [3072, 2048, 1024]},
        'decoder_ffn_embed_dim': {'count': 6, 'vars': [3072, 2048, 1024]},
        'decoder_layer_num': {'count': 1, 'vars': [6, 5, 4, 3, 2, 1]},
        'encoder_self_attention_heads': {'count': 6, 'vars': [8, 4]},
        'decoder_self_attention_heads': {'count': 6, 'vars': [8, 4]},
        'decoder_ende_attention_heads': {'count': 6, 'vars': [8, 4]},
        'decoder_arbitrary_ende_attn': {'count': 6, 'vars': [-1, 1, 2]},
    },
    'bert_base_sst2': {
        'num_layers': {'count': 1, 'vars': [6, 7, 8, 9, 10, 11, 12]},
        'num_attention_heads': {'count': 12, 'vars': [6, 8, 10, 12]},
        'intermediate_size': {'count': 12, 'vars': [1024, 2048, 3072]},
    },
}

EVALUATION_INTERFACE = {
    'ofa_resnet50': EvaluationInterfaceOFAResNet50,
    'ofa_mbv3_d234_e346_k357_w1.0': EvaluationInterfaceOFAMobileNetV3,
    'ofa_mbv3_d234_e346_k357_w1.2': EvaluationInterfaceOFAMobileNetV3,
    'ofa_proxyless_d234_e346_k357_w1.3': EvaluationInterfaceOFAMobileNetV3,
    'transformer_lt_wmt_en_de': EvaluationInterfaceTransformerLT,
    'bert_base_sst2': EvaluationInterfaceBertSST2,
}

LINAS_INNERLOOP_EVALS = {
    'ofa_resnet50': 5000,
    'ofa_mbv3_d234_e346_k357_w1.0': 20000,
    'ofa_mbv3_d234_e346_k357_w1.2': 20000,
    'ofa_proxyless_d234_e346_k357_w1.3': 20000,
    'transformer_lt_wmt_en_de': 10000,
    'bert_base_sst2': 10000,
}

SUPERNET_TYPE = {
    'image_classification': [
        'ofa_resnet50',
        'ofa_mbv3_d234_e346_k357_w1.0',
        'ofa_mbv3_d234_e346_k357_w1.2',
        'ofa_proxyless_d234_e346_k357_w1.3',
    ],
    'machine_translation': ['transformer_lt_wmt_en_de'],
    'text_classification': ['bert_base_sst2'],
    'recommendation': [],
}

SUPERNET_METRICS = {
    'ofa_resnet50': ['params', 'latency', 'macs', 'accuracy_top1'],
    'ofa_mbv3_d234_e346_k357_w1.0': ['params', 'latency', 'macs', 'accuracy_top1'],
    'ofa_mbv3_d234_e346_k357_w1.2': ['params', 'latency', 'macs', 'accuracy_top1'],
    'ofa_proxyless_d234_e346_k357_w1.3': ['params', 'latency', 'macs', 'accuracy_top1'],
    'transformer_lt_wmt_en_de': ['latency', 'macs', 'params', 'bleu'],
    'bert_base_sst2': ['latency', 'macs', 'params', 'accuracy_sst2'],
}


SEARCH_ALGORITHMS = ['linas', 'evolutionary', 'random']
