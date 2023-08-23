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

"""
BSD 3-Clause License
Copyright (c) Soumith Chintala 2016, 
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules_supernetwork import (
    Conv2dNormActivation,
    SuperEmbedding,
    SuperLayerNorm,
    SuperLinear,
    SuperMultiheadAttention,
    SuperSelfAttentionOutput,
)


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim, mlp_dim, dropout):
        super().__init__()
        self.linear_1 = SuperLinear(super_in_dim=in_dim, super_out_dim=mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = SuperLinear(super_in_dim=mlp_dim, super_out_dim=in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)

        # Elastic Parameters
        self.sample_hidden_size = None
        self.sample_intermediate_size = None

    def set_sample_config(self, vit_hidden_size, vit_intermediate_size):
        self.sample_hidden_size = vit_hidden_size
        self.sample_intermediate_size = vit_intermediate_size

        self.linear_1.set_sample_config(
            self.sample_hidden_size,
            self.sample_intermediate_size,
        )

        self.linear_2.set_sample_config(
            self.sample_intermediate_size,
            self.sample_hidden_size,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer=partial(SuperLayerNorm, eps=1e-6),  # partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SuperMultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.attention_output = SuperSelfAttentionOutput(
            hidden_dim,
            layer_norm_eps=1e-6,
            hidden_dropout_prob=dropout,
            num_attention_heads=num_heads,
        )
        self.dropout = nn.Dropout(dropout)

        # Intermediate MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        # Elastic Parameters
        self.sample_hidden_size = None
        self.sample_intermediate_size = None
        self.sample_num_attention_heads = None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        x = self.ln_1(input)
        # Self Attention + MLP
        x, _ = self.self_attention(x, output_attentions=True)
        x = self.attention_output(x, input)

        # Intermediate MLP
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

    def set_sample_config(
        self,
        vit_num_attention_heads,
        vit_sample_hidden_size,
        vit_sample_intermediate_size,
    ):
        self.sample_hidden_size = vit_sample_hidden_size
        self.sample_intermediate_size = vit_sample_intermediate_size
        self.sample_num_attention_heads = vit_num_attention_heads
        self.self_attention.set_sample_config(
            self.sample_hidden_size,
            self.sample_num_attention_heads,
        )
        self.attention_output.set_sample_config(
            self.sample_hidden_size,
            self.sample_num_attention_heads,
        )
        self.mlp.set_sample_config(
            self.sample_hidden_size,
            self.sample_intermediate_size,
        )
        # Update LayerNorm Size
        self.ln_1.set_sample_config(self.sample_hidden_size)
        self.ln_2.set_sample_config(self.sample_hidden_size)


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer=partial(SuperLayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = SuperEmbedding(seq_length, hidden_dim)
        self.register_buffer("position_ids", torch.arange(seq_length).expand((1, -1)))
        self.dropout = nn.Dropout(dropout)
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

        # Elastic Parameters
        self.sample_config = None
        self.sample_num_layer = None
        self.sample_hidden_size = None

    def forward(self, input: torch.Tensor, position_ids=None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        seq_length = input.shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        input = input + self.pos_embedding(position_ids)
        x = self.dropout(input)
        for i in range(self.sample_num_layer):
            x = self.layers[i](x)
        x = self.ln(x)
        return x

    def set_sample_config(self, sample_config):
        self.sample_config = sample_config
        self.sample_num_layer = sample_config["num_layers"]
        self.sample_hidden_size = sample_config["vit_hidden_sizes"]

        for i in range(self.sample_num_layer):
            tmp_layer = self.layers[i]
            sample_intermediate_size = sample_config['vit_intermediate_sizes'][i]
            sample_num_attention_heads = sample_config['num_attention_heads'][i]
            tmp_layer.set_sample_config(sample_num_attention_heads, self.sample_hidden_size, sample_intermediate_size)

        self.pos_embedding.set_sample_config(self.sample_hidden_size)
        self.ln.set_sample_config(self.sample_hidden_size)


class SuperViT(nn.Module):
    """ViT Baseline adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py"""

    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=1000,
        representation_size=None,
        norm_layer=partial(SuperLayerNorm, eps=1e-6),
        conv_stem_configs=None,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

        self.sample_config = None

    def _process_input(self, x: torch.Tensor):
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.heads(x)
        return x

    def set_sample_config(self, sample_config):
        self.sample_config = sample_config
        self.encoder.set_sample_config(sample_config)
