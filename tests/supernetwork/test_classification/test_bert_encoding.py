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

from dynast.supernetwork.supernetwork_registry import SUPERNET_PARAMETERS
from dynast.supernetwork.text_classification.bert_encoding import BertSST2Encoding


def test_BertSST2Encoding_random_sample():
    bertss2encoding = BertSST2Encoding(
        param_dict=SUPERNET_PARAMETERS['bert_base_sst2'],
        seed=42,
    )

    # fmt: off
    assert [5, 0, 0, 2, 1, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 2, 2, 2, 2, 1] == bertss2encoding.random_sample()
    # fmt: on


def test_bertss2encoding_translate2param() -> None:
    bertss2encoding = BertSST2Encoding(
        param_dict=SUPERNET_PARAMETERS['bert_base_sst2'],
        seed=42,
    )

    oh = bertss2encoding.random_sample()
    subcfg = bertss2encoding.translate2param(oh)

    assert subcfg == {
        'num_layers': [11],
        'num_attention_heads': [6, 6, 10, 8, 8, 8, 6, 6, 12, 6, 6, 6],
        'intermediate_size': [1024, 1024, 3072, 3072, 1024, 3072, 1024, 3072, 3072, 3072, 3072, 2048],
    }


def test_bertss2encoding_translate2param():
    bertss2encoding = BertSST2Encoding(
        param_dict=SUPERNET_PARAMETERS['bert_base_sst2'],
        seed=42,
    )

    subcfg = {
        'num_layers': [11],
        'num_attention_heads': [6, 6, 10, 8, 8, 8, 6, 6, 12, 6, 6, 6],
        'intermediate_size': [1024, 1024, 3072, 3072, 1024, 3072, 1024, 3072, 3072, 3072, 3072, 2048],
    }

    # fmt: off
    assert [11, 6, 6, 10, 8, 8, 8, 6, 6, 12, 6, 6, 0, 1024, 1024, 3072, 3072, 1024, 3072, 1024, 3072, 3072, 3072, 3072, 0] == bertss2encoding.onehot_custom(
        subcfg,
        provide_onehot=False,
    )
    # fmt: on
