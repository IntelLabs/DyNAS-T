# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.


from dynast.supernetwork.image_classification.ofa.ofa_encoding import OFAMobileNetV3Encoding, OFAResNet50Encoding
from dynast.supernetwork.image_classification.ofa.ofa_interface import (
    EvaluationInterfaceOFAMobileNetV3,
    EvaluationInterfaceOFAResNet50,
)
from dynast.supernetwork.machine_translation.transformer_encoding import TransformerLTEncoding
from dynast.supernetwork.machine_translation.transformer_interface import EvaluationInterfaceTransformerLT

SUPERNET_ENCODING = {
    'ofa_resnet50': OFAResNet50Encoding,
    'ofa_mbv3_d234_e346_k357_w1.0': OFAMobileNetV3Encoding,
    'ofa_mbv3_d234_e346_k357_w1.2': OFAMobileNetV3Encoding,
    'ofa_proxyless_d234_e346_k357_w1.3': OFAMobileNetV3Encoding,
    'transformer_lt_wmt_en_de': TransformerLTEncoding,
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
}

EVALUATION_INTERFACE = {
    'ofa_resnet50': EvaluationInterfaceOFAResNet50,
    'ofa_mbv3_d234_e346_k357_w1.0': EvaluationInterfaceOFAMobileNetV3,
    'ofa_mbv3_d234_e346_k357_w1.2': EvaluationInterfaceOFAMobileNetV3,
    'ofa_proxyless_d234_e346_k357_w1.3': EvaluationInterfaceOFAMobileNetV3,
    'transformer_lt_wmt_en_de': EvaluationInterfaceTransformerLT,
}

LINAS_INNERLOOP_EVALS = {
    'ofa_resnet50': 5000,
    'ofa_mbv3_d234_e346_k357_w1.0': 20000,
    'ofa_mbv3_d234_e346_k357_w1.2': 20000,
    'ofa_proxyless_d234_e346_k357_w1.3': 20000,
    'transformer_lt_wmt_en_de': 10000,
}

SUPERNET_TYPE = {
    'image_classification': [
        'ofa_resnet50',
        'ofa_mbv3_d234_e346_k357_w1.0',
        'ofa_mbv3_d234_e346_k357_w1.2',
        'ofa_proxyless_d234_e346_k357_w1.3',
    ],
    'machine_translation': ['transformer_lt_wmt_en_de'],
    'recommendation': [],
}

SUPERNET_METRICS = {
    'ofa_resnet50': ['params', 'latency', 'macs', 'acc'],
    'ofa_mbv3_d234_e346_k357_w1.0': ['params', 'latency', 'macs', 'acc'],
    'ofa_mbv3_d234_e346_k357_w1.2': ['params', 'latency', 'macs', 'acc'],
    'ofa_proxyless_d234_e346_k357_w1.3': ['params', 'latency', 'macs', 'acc'],
    'transformer_lt_wmt_en_de': ['latency', 'macs', 'params', 'bleu'],
}


SEARCH_ALGORITHMS = ['linas', 'evolutionary', 'random']
