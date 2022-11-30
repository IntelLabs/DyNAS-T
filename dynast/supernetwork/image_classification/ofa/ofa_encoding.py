# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

import numpy as np

from dynast.search.encoding import EncodingBase


class OFAMobileNetV3Encoding(EncodingBase):
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):
        super().__init__(param_dict, verbose, seed)

    def construct_maps(self, keys):
        d = dict()
        keys = list(set(keys))
        for k in keys:
            if k not in d:
                d[k] = len(list(d.keys()))
        return d

    def onehot_custom(self, ks_list, ex_list, d_list):

        ks_map = self.construct_maps(keys=(3, 5, 7))
        ex_map = self.construct_maps(keys=(3, 4, 6))
        dp_map = self.construct_maps(keys=(2, 3, 4))

        # This function converts a network config to a feature vector (128-D).
        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

        # convert to onehot
        ks_onehot = [0 for _ in range(60)]
        ex_onehot = [0 for _ in range(60)]

        for i in range(20):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + ex_map[ex_list[i]]] = 1

        return np.array(ks_onehot + ex_onehot)


class OFAResNet50Encoding(EncodingBase):
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):
        super().__init__(param_dict, verbose, seed)
