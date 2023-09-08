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

from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn


class _DepthParser(ABC):
    def __init__(self, supernet_depth):
        super().__init__()
        self.supernet_depth = supernet_depth

    @abstractmethod
    def blockwise_masks(self, subnet_depth) -> list:
        raise NotImplementedError()

    @abstractmethod
    def layerwise_masks(self, subnet_depth) -> list:
        raise NotImplementedError()

    @abstractmethod
    def regex_block_names(self, subnet_depth, block_keyword) -> list:
        raise NotImplementedError()

    @abstractmethod
    def regex_layer_names(self, subnet_model) -> list:
        raise NotImplementedError()

    def block_parse(self, subnet_depth, block_keyword='blocks'):
        if block_keyword is None:
            return self.blockwise_masks(subnet_depth)
        else:
            return self.blockwise_masks(subnet_depth), self.regex_block_names(subnet_depth, block_keyword)

    def layer_parse(self, subnet_depth, subnet_model=None):
        if subnet_model is None:
            return self.layerwise_masks(subnet_depth)
        else:
            return self.layerwise_masks(subnet_depth), self.regex_layer_names(subnet_model)


class OFA_ResNet50_DepthParser(_DepthParser):
    def __init__(self, supernet_depth=[2] * 5, base_blocks=[2, 2, 4, 2]):
        super().__init__(supernet_depth=supernet_depth)
        self.base_blocks = base_blocks

    def blockwise_masks(self, subnet_depth):
        base_blocks, supernet_depth, subnet_depth = self.base_blocks, self.supernet_depth[1:], subnet_depth[1:]
        assert len(base_blocks) == len(supernet_depth) == len(subnet_depth)
        stage_num, full_depth = len(supernet_depth), sum(base_blocks) + sum(supernet_depth)

        masks = []
        # ---------- OFA ResNet50 block parse ----------
        # # parse the network depth
        for stage_i in range(stage_num):
            for _ in range(base_blocks[stage_i]):
                masks.append(True)
            for block_i in range(supernet_depth[stage_i]):
                if block_i < subnet_depth[stage_i]:
                    masks.append(True)
                else:
                    masks.append(False)
        assert len(masks) == full_depth
        # ------------------------------------------------
        return masks

    def regex_block_names(self, subnet_depth, block_keyword):
        base_blocks, subnet_depth = self.base_blocks, subnet_depth[1:]
        assert len(base_blocks) == len(subnet_depth)

        regex_module_names = []
        for block_i in range(sum(base_blocks) + sum(subnet_depth)):
            regex_module_names.append(f"^({block_keyword}).{block_i}\..*$")
        return regex_module_names

    def layerwise_masks(self, subnet_depth):
        input_stem_depth = subnet_depth[0]
        base_blocks, supernet_depth, subnet_depth = self.base_blocks, self.supernet_depth[1:], subnet_depth[1:]
        assert len(base_blocks) == len(supernet_depth) == len(subnet_depth)
        stage_num, full_depth = len(supernet_depth), 61

        masks = []
        # ---------- OFA ResNet50 layer parse ----------
        # # parse the network depth
        if input_stem_depth == 2:
            masks.extend([True, True, True])
        else:
            masks.extend([True, True, False])

        for stage_i in range(stage_num):
            masks.extend([True, True, True, True])
            for _ in range(base_blocks[stage_i] - 1):
                masks.extend([True, True, True])
            for block_i in range(supernet_depth[stage_i]):
                if block_i < subnet_depth[stage_i]:
                    masks.extend([True, True, True])
                else:
                    masks.extend([False, False, False])
        assert len(masks) == full_depth
        # -------------------------------------------------
        return masks

    def regex_layer_names(self, subnet_model):
        regex_module_names = []
        for name, module in subnet_model.named_modules():
            if type(module) in (nn.modules.conv.Conv2d,) and name.endswith('.conv'):
                regex_module_names.append(f'^({name})$')
        return regex_module_names


class DepthParser:
    def __new__(self, supernet, **kwargs):
        return OFA_ResNet50_DepthParser(**kwargs)
