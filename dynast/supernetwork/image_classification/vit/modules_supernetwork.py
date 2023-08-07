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
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, super_embed_dim, padding_idx=None, *args, **kwargs):
        super().__init__(num_embeddings, super_embed_dim, padding_idx, *args, **kwargs)

        self.super_embed_dim = super_embed_dim

        self.sample_embed_dim = None

        self.samples = {}
        self.reset_parameters()

        self.profiling = False

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        nn.init.constant_(self.weight[self.padding_idx], 0)

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def _sample_parameters(self):
        weight = self.weight[..., : self.sample_embed_dim]
        self.samples['weight'] = weight

        return self.samples

    def sample_parameters(self, resample=False):
        return self._sample_parameters() if self.profiling or resample else self.samples

    def sampled_weight(self):
        return self.sample_parameters()['weight']

    def forward(self, input):
        return F.embedding(
            input,
            self.sampled_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def profile(self, mode=True):
        self.profiling = mode

    def calc_sampled_param_num(self):
        return self.samples.numel()


class SuperLayerNorm(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim, eps):
        super().__init__(super_embed_dim, eps)

        self.super_embed_dim = super_embed_dim

        self.sample_embed_dim = None
        self.samples = {}

        self.profiling = False

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[: self.sample_embed_dim]
        self.samples['bias'] = self.bias[: self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(
            x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps
        )

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def profile(self, mode=True):
        self.profiling = mode


class SuperLinear(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}
        super().reset_parameters()

        self.profiling = False

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias'])

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def profile(self, mode=True):
        self.profiling = mode


def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]

    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        norm_layer=torch.nn.BatchNorm2d,
        activation_layer=torch.nn.ReLU,
        dilation=1,
        inplace=True,
        bias=None,
        conv_layer=torch.nn.Conv2d,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        norm_layer=torch.nn.BatchNorm2d,
        activation_layer=torch.nn.ReLU,
        dilation=1,
        inplace=True,
        bias=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )


class SuperSelfAttentionOutput(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps, hidden_dropout_prob, num_attention_heads):
        super().__init__()
        self.dense = SuperLinear(hidden_size, hidden_size)
        self.LayerNorm = SuperLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.sample_hidden_size = None
        self.sample_head_num = None

        self.origin_num_attention_heads = num_attention_heads
        self.origin_attention_head_size = int(hidden_size / num_attention_heads)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def set_sample_config(self, vit_hidden_size, vit_head_num):
        self.sample_hidden_size = vit_hidden_size
        self.sample_head_num = vit_head_num
        self.sample_all_head_size = self.origin_attention_head_size * self.sample_head_num

        self.dense.set_sample_config(self.sample_all_head_size, self.sample_hidden_size)
        self.LayerNorm.set_sample_config(self.sample_hidden_size)


class SuperMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, batch_first):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads)
            )
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)  # 384 / 6 = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 6 * 64 = 384

        self.query = SuperLinear(hidden_size, self.all_head_size)
        self.key = SuperLinear(hidden_size, self.all_head_size)
        self.value = SuperLinear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

        self.batch_first = batch_first

        # Elastic Parameters
        self.sample_hidden_size = None
        self.sample_num_attention_heads = None
        self.sample_attention_head_size = None
        self.sample_all_head_size = None

        self.super_hidden_size = hidden_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.sample_num_attention_heads, self.sample_attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.sample_all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def set_sample_config(self, vit_hidden_size, vit_head_num):
        self.sample_hidden_size = vit_hidden_size
        self.sample_num_attention_heads = vit_head_num

        self.sample_attention_head_size = self.attention_head_size  # 64
        self.sample_all_head_size = self.sample_num_attention_heads * self.sample_attention_head_size  # 64 x 4 = 256

        self.query.set_sample_config(sample_in_dim=self.sample_hidden_size, sample_out_dim=self.sample_all_head_size)
        self.key.set_sample_config(sample_in_dim=self.sample_hidden_size, sample_out_dim=self.sample_all_head_size)
        self.value.set_sample_config(sample_in_dim=self.sample_hidden_size, sample_out_dim=self.sample_all_head_size)
