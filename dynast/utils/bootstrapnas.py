# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

import torch
from bnas_resnets import BootstrapNASResnet50

from dynast.quantization import quantize_ov, validate
from dynast.utils import samples_to_batch_multiply
from dynast.utils.datasets import ImageNet
from dynast.utils.nn import count_parameters, get_gflops, reset_bn
from dynast.utils.ov import benchmark_openvino

SUPERNET_PARAMETERS = {
    'd': {'count': 5, 'vars': [0, 1]},
    'e': {'count': 12, 'vars': [0.2, 0.25]},
    'w': {'count': 6, 'vars': [0, 1, 2]},
}


class BNASRunner:
    def __init__(
        self,
        supernet,
        acc_predictor=None,
        latency_predictor=None,
        macs_predictor=None,
        bn_samples=4000,
        batch_size=128,
        resolution=224,
        cores=56,
    ):
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.latency_predictor = latency_predictor
        self.macs_predictor = macs_predictor
        self.bn_samples = bn_samples
        self.batch_size = batch_size
        self.resolution = resolution
        self.cores = cores
        self.img_size = (self.batch_size, 3, self.resolution, self.resolution)

    def estimate_accuracy_top1(self, subnet_cfg):
        assert self.acc_predictor, 'Please provide `acc_predictor` when creating BNASRunner object.'
        top1 = self.acc_predictor.predict_single(subnet_cfg)
        return top1

    def estimate_latency(self, subnet_cfg):
        assert self.acc_predictor, 'Please provide `latency_predictor` when creating BNASRunner object.'
        lat = self.latency_predictor.predict_single(subnet_cfg)
        return lat

    def validate_subnet(self, subnet_cfg):
        top1, top5 = validate()

        self.supernet.set_active_subnet(d=subnet_cfg['d'], e=subnet_cfg['e'], w=subnet_cfg['w'])
        subnet = self.supernet.get_active_subnet()
        gflops = get_gflops(subnet)
        model_params = count_parameters(subnet)

        return top1, top5, gflops, model_params

    def benchmark_subnet(self):
        latency_ov, fps_ov = benchmark_openvino(
            self.img_size,
            cores=self.cores,
            is_quantized=True,
        )

        return latency_ov

    def quantize(self, subnet_cfg):
        self.supernet.set_active_subnet(d=subnet_cfg['d'], e=subnet_cfg['e'], w=subnet_cfg['w'])
        subnet = self.supernet.get_active_subnet()

        train_dataloader = ImageNet.train_dataloader(batch_size=self.batch_size)

        reset_bn(
            model=subnet,
            num_samples=samples_to_batch_multiply(self.bn_samples, self.batch_size),
            train_dataloader=train_dataloader,
        )

        quantize_ov(model=subnet, img_size=self.img_size)


def get_supernet(path='supernet/torchvision_resnet50_supernet.pth', device='cpu'):
    supernet = BootstrapNASResnet50(depth_list=[0, 1], expand_ratio_list=[0.2, 0.25], width_mult_list=[0.65, 0.8, 1.0])

    init = torch.load(path, map_location=torch.device(device))['state_dict']
    supernet.load_state_dict(init)
    return supernet
