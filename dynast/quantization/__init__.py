# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

import os
from typing import Tuple

import torch

from dynast.utils import get_hostname, samples_to_batch_multiply
from dynast.utils.datasets import ImageNet
from dynast.utils.nn import validate_classification
from dynast.utils.ov import load_openvino, save_openvino, save_ov_quantized


def quantize_ov(
    model: torch.nn.Module,
    img_size: Tuple[int, int, int, int],
    experiment_name: str = None,
    quant_policy: str = 'DefaultQuantization',
    stat_subset_size: int = None,
) -> None:
    """Converts Torch model to OpenVINO Quantized model

    Arguments:
    ----------
    * `model`: Torch model
    * `img_size`: input tensor size (batch size, channels, resolution, resolution)
    * `experiment_name`: name which will be used to identify results of this experiment. If `None`
      it will be automatically sety to `${HOSTNAME}_dynast_eval`.
    * `quant_policy`: identifies which quantization policy should be used. Please refer to
      `dynast/quantization/policy.py` for the list of available options.
    * `stat_subset_size`: how many samples to use when calibrating quantized model. It is encouraged to
      set it to at least 300 and to `stat_subset_size` be divisable by batch size. If not set proper value
      will be assigned automatically based on these two rules.

    Returns:
    --------
    * None
    """
    if not experiment_name:
        experiment_name = '{}_dynast_eval'.format(get_hostname())
    if not stat_subset_size:
        stat_subset_size = samples_to_batch_multiply(300, img_size[0])

    folder_name = os.path.expanduser('/store/.torch/{}'.format(experiment_name))
    ov_model_dir = os.path.join(folder_name, 'ov_model')

    save_openvino(model, img_size, ov_model_dir, experiment_name)

    save_ov_quantized(
        tmp_folder=ov_model_dir,
        model_name=experiment_name,
        quant_policy=quant_policy,
        stat_subset_size=stat_subset_size,
    )


def validate(
    experiment_name: str = None,
    test_size: int = None,
    batch_size: int = 128,
) -> Tuple[float, float]:
    """Evaluates quantized model.

    Arguments:
    * `experiment_name`: name which will be used to identify results of this experiment. If `None`
      it will be automatically set to `${HOSTNAME}_dynast_eval`.
    * `test_size`: How many batches of data to use when validaing. Set to `None` to use all data.
    * `batch_size`: Input batch size for validation.

    Returns:
    --------
    * Top1 accuracy, Top5 accuracy
    """

    if not experiment_name:
        experiment_name = '{}_dynast_eval'.format(get_hostname())

    folder_name = os.path.expanduser('/store/.torch/{}'.format(experiment_name))
    ov_model_dir = os.path.join(folder_name, 'ov_model')

    subnet_ov_int8 = load_openvino(folder=ov_model_dir, name=experiment_name, is_quantized=True)

    loss_ov_quant, top1_ov_quant, top5_ov_quant = validate_classification(
        model=subnet_ov_int8,
        is_openvino=True,
        test_size=test_size,
        batch_size=batch_size,
        data_loader=ImageNet.validation_dataloader(batch_size=batch_size),
    )
    return top1_ov_quant, top5_ov_quant
