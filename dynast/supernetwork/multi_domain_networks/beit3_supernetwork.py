# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn
import copy
from torchscale.architecture.encoder import Encoder
from torchscale.component.embedding import (
    PositionalEmbedding,
    TextEmbedding,
    VisionEmbedding,
)
from torchscale.component.multiway_network import MutliwayEmbedding


# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import numpy as np
import torch
import torch.nn as nn
from fairscale.nn import checkpoint_wrapper, wrap
#try:
#    from apex.normalization import FusedLayerNorm as LayerNorm
#except ModuleNotFoundError:
from torch.nn import LayerNorm

from torchscale.architecture.utils import init_bert_params
from torchscale.component.droppath import DropPath
#from torchscale.component.feedforward_network import FeedForwardNetwork, make_experts
#from torchscale.component.multihead_attention import MultiheadAttention
#rom torchscale.component.multiway_network import MultiwayWrapper, set_split_position
from torchscale.component.relative_position_bias import RelativePositionBias
from torchscale.component.xmoe.moe_layer import MOELayer
from torchscale.component.xpos_relative_position import XPOS
import torch.nn.functional as F

from torchscale.component.xmoe.routing import Top1Gate, Top2Gate
from .modules_supernetwork import LinearSuper,LayerNormSuper
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)
    
    def set_sample_config(self,sample_embed_size, sample_all_head_size):
        self.A.set_sample_config(sample_embed_size, sample_all_head_size)
        self.B.set_sample_config(sample_embed_size, sample_all_head_size)
    
    def set_sample_config_layernorm(self,sample_all_head_size):
         self.A.set_sample_config(sample_all_head_size)
         self.B.set_sample_config(sample_all_head_size)


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = self.get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_rng_state(self):
        state = {"torch_rng_state": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
        return state

    def set_rng_state(self, state):
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng_state"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.set_rng_state(self.rng_state)


def make_experts(args, embed_dim, expert_ffn_dim):
    world_size = (
        1
        if not torch.distributed.is_initialized()
        else torch.distributed.get_world_size()
    )
    expert_list = []
    ddp_rank = args.ddp_rank
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert (
            args.moe_expert_count % world_size == 0
        ), f"{args.moe_expert_count}, {world_size}"
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(
                    FeedForwardNetwork(
                        embed_dim,
                        expert_ffn_dim,
                        args.activation_fn,
                        args.dropout,
                        args.activation_dropout,
                        args.layernorm_eps,
                        args.subln,
                    )
                )
    else:
        assert (
            world_size % args.moe_expert_count == 0
        ), f"{world_size}, {args.moe_expert_count}"

        with set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            expert_list.append(
                FeedForwardNetwork(
                    embed_dim,
                    expert_ffn_dim,
                    args.activation_fn,
                    args.dropout,
                    args.activation_dropout,
                    args.layernorm_eps,
                    args.subln,
                )
            )
    experts = nn.ModuleList(expert_list)
    return experts


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = LinearSuper(self.embed_dim, ffn_dim)
        self.fc2 = LinearSuper(ffn_dim, self.embed_dim)
        self.ffn_layernorm = LayerNormSuper(ffn_dim, eps=layernorm_eps) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x
    
    def set_sample_config(self,sample_embed_size,sample_ffn_size):
        self.fc1.set_sample_config(sample_embed_size,sample_ffn_size)
        self.fc2.set_sample_config(sample_ffn_size,sample_embed_size)
        self.ffn_layernorm.set_sample_config(sample_ffn_size)

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, LinearSuper(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, LinearSuper(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, LinearSuper(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args,LinearSuper(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNormSuper(self.embed_dim, eps=args.layernorm_eps))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.sample_head_num, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.sample_head_num, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len,self.sample_head_num, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.sample_head_num, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.sample_head_num, src_len, self.head_dim)
        v = v.reshape(bsz * self.sample_head_num, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.sample_head_num, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.sample_head_num, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.sample_head_num, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.sample_head_num, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
       
        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.sample_head_num, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.sample_head_num, tgt_len, src_len)

        if rel_pos is not None:
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz,  self.sample_all_head_size).transpose(0, 1)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz,  self.sample_head_num, tgt_len, src_len
        ).transpose(1, 0)

        return attn, attn_weights
    
    def set_sample_config(self, sample_embed_size, sample_head_num):
        self.sample_head_num = sample_head_num
        self.sample_embed_size = sample_embed_size
        
        self.sample_all_head_size = sample_head_num * self.head_dim
        self.q_proj.set_sample_config(sample_embed_size, self.sample_all_head_size)
        self.k_proj.set_sample_config(sample_embed_size, self.sample_all_head_size)
        self.v_proj.set_sample_config(sample_embed_size, self.sample_all_head_size)
        self.inner_attn_ln.set_sample_config_layernorm(self.sample_all_head_size)
        self.out_proj.set_sample_config(self.sample_all_head_size,sample_embed_size)





class EncoderLayer(nn.Module):
    def __init__(self, args, depth, is_moe_layer=False, is_encoder_decoder=False):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
        self.dropout_module = torch.nn.Dropout(args.dropout)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.encoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.normalize_before = args.encoder_normalize_before
        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.encoder_ffn_embed_dim

        if not self.is_moe_layer:
            self.ffn = MultiwayWrapper(
                args,
                self.build_ffn(
                    self.embed_dim,
                    self.args,
                ),
            )
        else:
            assert not self.args.multiway
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim)
            self.moe_layer = MOELayer(gate, experts, args)
        self.final_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))

        if args.deepnorm:
            if is_encoder_decoder:
                self.alpha = (
                    math.pow(
                        math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
                    )
                    * 0.81
                )
            else:
                self.alpha = math.pow(2.0 * args.encoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
            args.subln,
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(self, x, encoder_padding_mask, attn_mask=None, rel_pos=None, multiway_split_position=None, incremental_state=None):
        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x = x.transpose(0, 1)
            x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, l_aux
    
    def set_sample_config(self,sample_embed_size, sample_ffn_size, sample_head_num):
        self.self_attn.set_sample_config(sample_embed_size, sample_head_num)
        self.ffn.set_sample_config(sample_embed_size,sample_ffn_size)


class Encoder(nn.Module):
    def __init__(
        self,
        args,
        embed_tokens=None,
        embed_positions=None,
        output_projection=None,
        is_encoder_decoder=False,
        **kwargs
    ):
        self.args = args
        super().__init__(**kwargs)

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.encoder_embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        if (
            output_projection is None
            and not is_encoder_decoder
            and not args.no_output_layer
            and args.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = MultiwayWrapper(
                args, LayerNorm(embed_dim, eps=args.layernorm_eps), dim=1
            )
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.encoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_encoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                    is_encoder_decoder=is_encoder_decoder,
                )
            )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before and args.normalize_output:
            self.layer_norm = MultiwayWrapper(args, LayerNorm(embed_dim, eps=args.layernorm_eps))
        else:
            self.layer_norm = None

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.encoder_attention_heads,
            )
        else:
            self.relative_position = None

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = (
                    math.pow(
                        math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
                    )
                    / 1.15
                )
            else:
                init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if args.subln:
            if is_encoder_decoder:
                init_scale = math.sqrt(
                    math.log(3 * args.decoder_layers)
                    * math.log(2 * args.encoder_layers)
                    / 3
                )
            else:
                init_scale = math.sqrt(math.log(args.encoder_layers * 2))
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.mul_(init_scale)

    def build_output_projection(
        self,
        args,
    ):
        if args.share_encoder_input_output_embed:
            assert args.encoder_embedding_type == "language"
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.encoder_embed_dim, args.vocab_size, bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.encoder_embed_dim**-0.5
            )
        return output_projection

    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = EncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def forward_embedding(
        self,
        src_tokens,
        token_embedding=None,
        positions=None,
    ):
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            if src_tokens is not None:
                x = embed + self.embed_positions(src_tokens, positions=positions)
            else:
                x = embed + self.embed_positions(x, positions=positions)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        encoder_padding_mask=None,
        attn_mask=None,
        return_all_hiddens=False,
        token_embeddings=None,
        multiway_split_position=None,
        features_only=False,
        incremental_state=None,
        positions=None,
        **kwargs
    ):
        assert src_tokens is not None or token_embeddings is not None

        if encoder_padding_mask is None:
            if src_tokens is not None:
                encoder_padding_mask = torch.zeros_like(
                    src_tokens, device=src_tokens.device
                ).bool()
            else:
                encoder_padding_mask = torch.zeros(
                    [token_embeddings.size(0), token_embeddings.size(1)],
                    device=token_embeddings.device,
                ).bool()

        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings, positions)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        rel_pos_bias = None
        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(
                batch_size=x.size(0), qlen=x.size(1), klen=x.size(1)
            )

        # incremental_state is not None during inference if we use the bidirectional encoder as a generator as in s2s-ft (https://arxiv.org/abs/2110.13640)
        l_aux = []
        for idx, layer in enumerate(self.layers):
            if idx >= self.sample_num_layer:
                break;
            else:
                x, l_aux_i = layer(
                    x,
                    encoder_padding_mask=encoder_padding_mask if incremental_state is None else None,
                    attn_mask=attn_mask,
                    rel_pos=rel_pos_bias,
                    multiway_split_position=multiway_split_position,
                    incremental_state=incremental_state[idx] if incremental_state is not None else None,
                )
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)
                l_aux.append(l_aux_i)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only and self.output_projection is not None:
            x = self.output_projection(x)

        return {
            "encoder_out": x,
            "encoder_embedding": encoder_embedding,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
            "l_aux": l_aux,
        }

    def set_sample_config(self,sample_config):
        self.sample_config = sample_config
        self.sample_num_layer = sample_config["num_layers"]
        self.sample_embed_size = sample_config["embed_size"]

        for i,layer in enumerate(self.layers):
            if i>=self.sample_num_layer:
                break;
            else:
                tmp_layer = layer

                sample_ffn_size = sample_config['ffn_size'][i]
                sample_head_num = sample_config['head_num'][i]

                tmp_layer.set_sample_config(self.sample_embed_size, sample_ffn_size, sample_head_num)





class BEiT3(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert args.vocab_size > 0
        assert not args.share_encoder_input_output_embed
        self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        self.vision_embed = VisionEmbedding(
            args.img_size,
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim),
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions,
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out
    
    def set_sample_config(self,sample_config):
        self.encoder.set_sample_config(sample_config)




class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    


class BEiT3ForImageClassification(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.fc_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def forward(self, image, **kwargs):
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        t = x[:, 1:, :]
        cls_x = self.fc_norm(t.mean(1))
        return self.head(cls_x)
    

    def set_sample_config(self,sample_config):
        self.beit3.set_sample_config(sample_config)


