# Copyright 2018- The Hugging Face team. All rights reserved.
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

from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEncoder,
    BertForSequenceClassification,
    BertIntermediate,
    BertLayer,
    BertModel,
    BertOutput,
    BertPooler,
    BertSelfAttention,
    BertSelfOutput,
)

from ..machine_translation.modules_supernetwork import LinearSuper


class BertSupernetSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.query = LinearSuper(config.hidden_size, self.all_head_size)
        self.key = LinearSuper(config.hidden_size, self.all_head_size)
        self.value = LinearSuper(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.subnet_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def set_sample_config(self, sub_layer_config):
        self.subnet_attention_heads = sub_layer_config['num_attention_heads']
        self.subnet_hidden_size = sub_layer_config['hidden_size']
        self.all_head_size = self.attention_head_size * self.subnet_attention_heads

        self.query.set_sample_config(self.subnet_hidden_size, self.all_head_size)
        self.key.set_sample_config(self.subnet_hidden_size, self.all_head_size)
        self.value.set_sample_config(self.subnet_hidden_size, self.all_head_size)


class BertSupernetSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)

        self.supernet_attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.dense = LinearSuper(config.hidden_size, config.hidden_size)

    def set_sample_config(self, sub_layer_config):
        subnet_all_head_size = self.supernet_attention_head_size * sub_layer_config['num_attention_heads']
        self.dense.set_sample_config(subnet_all_head_size, sub_layer_config['hidden_size'])


class BertSupernetAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertSupernetSelfAttention(config)
        self.output = BertSupernetSelfOutput(config)

    def set_sample_config(self, sub_layer_config):
        self.self.set_sample_config(sub_layer_config)
        self.output.set_sample_config(sub_layer_config)


class BertSupernetIntermediate(BertIntermediate):
    def __init__(self, config):
        super().__init__(config)
        self.dense = LinearSuper(config.hidden_size, config.intermediate_size)

    def set_sample_config(self, sub_layer_config):
        self.dense.set_sample_config(sub_layer_config['hidden_size'], sub_layer_config['intermediate_size'])


class BertSupernetOutput(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = LinearSuper(config.intermediate_size, config.hidden_size)

    def set_sample_config(self, sub_layer_config):
        self.dense.set_sample_config(sub_layer_config['intermediate_size'], sub_layer_config['hidden_size'])


class BertSupernetLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertSupernetAttention(config)
        self.intermediate = BertSupernetIntermediate(config)
        self.output = BertSupernetOutput(config)

    def set_sample_config(self, sub_layer_config):
        self.attention.set_sample_config(sub_layer_config)
        self.intermediate.set_sample_config(sub_layer_config)
        self.output.set_sample_config(sub_layer_config)


class BertSupernetEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertSupernetLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        for layer_idx, layer_module in enumerate(self.layer):
            if layer_idx >= self.subnet_num_layers:
                break

            layer_head_mask = None
            past_key_value = None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
        )

    def set_sample_config(self, subnetwork_config):
        self.subnet_num_layers = subnetwork_config["num_layers"]
        for layer_idx in range(self.subnet_num_layers):
            sub_layer_config = {
                'intermediate_size': subnetwork_config['intermediate_size'][layer_idx],
                'num_attention_heads': subnetwork_config['num_attention_heads'][layer_idx],
                'hidden_size': subnetwork_config["hidden_size"],
            }
            self.layer[layer_idx].set_sample_config(sub_layer_config)


class BertSupernetPooler(BertPooler):
    def __init__(self, config):
        super().__init__(config)
        self.dense = LinearSuper(config.hidden_size, config.hidden_size)

    def set_sample_config(self, subnetwork_config):
        self.dense.set_sample_config(subnetwork_config["hidden_size"], subnetwork_config["hidden_size"])


class BertSupernetModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = BertSupernetEncoder(config)
        self.pooler = BertSupernetPooler(config)

    def set_sample_config(self, subnetwork_config):
        self.encoder.set_sample_config(subnetwork_config)
        self.pooler.set_sample_config(subnetwork_config)


class BertSupernetForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.bert = BertSupernetModel(config)
        self.classifier = LinearSuper(config.hidden_size, config.num_labels)

    def set_sample_config(self, subnetwork_config):
        self.bert.set_sample_config(subnetwork_config)
        self.subnetwork_hidden_size = subnetwork_config["hidden_size"]
        self.classifier.set_sample_config(self.subnetwork_hidden_size, self.num_labels)
