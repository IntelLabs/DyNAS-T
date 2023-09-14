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

import copy
import os
import time
from typing import List, Union

import neural_compressor
import torch
import yaml
from neural_compressor.experimental import Quantization


def default_policy() -> dict:
    policy = {
        'model': {
            'name': 'dynast_quantized_subnet',
            'framework': 'pytorch_fx',
        },
        'device': 'cpu',
        'quantization': {
            'approach': 'post_training_static_quant',
            'calibration': {
                'sampling_size': 2000,
            },
            'model_wise': {
                'activation': {'dtype': 'fp32'},
                'weight': {'dtype': 'fp32'},
            },
        },
    }
    return policy


def qparam_parse(
    observer_type: str,
    bit: str,
    mode: str,
    granularity: str,
) -> (str, str, str, str):
    '''
    Parse quantization parameters
    Parameters:
        observer_type: algorithm for calculating scale and zero_point.
                       possible options: {'minmax', 'kl'}
        bit: quantized data type.
             possible options: {'float32', 'qint8', 'quint8'}
        mode: mode to convert float point range to quantized range.
              possible options: {'symmetric', 'asymmetric'}
        granularity: ways to utilize the input data statistics.
                     possible options: {'perchannel', 'pertensor'}
    Return:
        (observer, dtype, qscheme, qgranularity)
    '''
    observer = {
        'minmax': 'minmax',
        'kl': 'kl',
    }
    dtype = {
        'float32': 'fp32',
        'qint8': 'int8',
        'quint8': 'uint8',
    }
    qscheme = {
        'symmetric': 'sym',
        'asymmetric': 'asym',
    }
    qgranularity = {
        'pertensor': 'per_tensor',
        'perchannel': 'per_channel',
    }

    return (observer[observer_type], dtype[bit], qscheme[mode], qgranularity[granularity])


def qconfig_parse(
    wobserver_type: str,
    wbit: str,
    wmode: str,
    wgranularity: str,
    aobserver_type: str,
    abit: str,
    amode: str,
    agranularity: str,
) -> dict:
    '''
    Parse quantization config
    Parameters:
        wobserver_type: weight observer
        wbit: weight data type
        wmode: weight mode
        wgranularity: weight granularity
        aobserver_type: activation observer
        abit: activation data type
        amode: activation mode
        agranularity: activation granularity
    Return:
        INC QConfig
    '''
    w_observer, w_dtype, w_scheme, w_granularity = qparam_parse(wobserver_type, wbit, wmode, wgranularity)
    a_observer, a_dtype, a_scheme, a_granularity = qparam_parse(aobserver_type, abit, amode, agranularity)

    qconfig = {
        'weight': {
            'algorithm': [w_observer],
            'dtype': [w_dtype],
            'scheme': [w_scheme],
            'granularity': [w_granularity],
        }
        if w_dtype != 'fp32'
        else {'dtype': ['fp32']},
        'activation': {
            'algorithm': [a_observer],
            'dtype': [a_dtype],
            'scheme': [a_scheme],
            'granularity': [a_granularity],
        }
        if a_dtype != 'fp32'
        else {'dtype': ['fp32']},
    }

    return qconfig


def _convert_dtype(w_bit, a_bit):
    if w_bit == 32:
        wbit = 'float32'
    elif w_bit == 8:
        wbit = 'qint8'
    else:
        raise Exception(
            f'Unsupported Weight Data Type: {w_bit}!' + 'Only support float32 / qint8 by specify 32 / 8 temporarily!'
        )

    if a_bit == 32:
        abit = 'float32'
    elif a_bit == 8:
        abit = 'quint8'
    else:
        raise Exception(
            f'Unsupported Activation Data Type: {a_bit}!'
            + 'Only support float32 / quint8 by specify 32 / 8 temporarily!'
        )

    return (wbit, abit)


def inc_qconfig_dict(
    q_weights_bit: Union[List[int], int],
    q_activations_bit: Union[List[int], int],
    q_weights_mode: Union[List[str], str],
    q_activations_mode: Union[List[str], str],
    q_weights_granularity: Union[List[str], str],
    q_activations_granularity: Union[List[str], str],
    regex_module_names: list = None,
):
    '''
    Parameters:
        q_weights_bit (list of int or int): weight data type
        q_activations_bit (list of int or int): activation data type
        q_weights_mode (list of str or str): weight mode
        q_activations_mode (list of str or str): activation mode
        q_weights_granularity (list of str or str): weight granularity
        q_activations_granularity (list of str or str): activation granularity
        regex_module_names (list): name list of sub-modules to be quantized. if None, then quantize all sub-modules using the same policy.
    Return:
        A customized QConfig dictionary that specify the quantization configure for each specified module.
    '''

    qconfig_dict = default_policy()

    if regex_module_names is not None:
        assert (
            len(regex_module_names)
            == len(q_weights_bit)
            == len(q_activations_bit)
            == len(q_weights_mode)
            == len(q_activations_mode)
            == len(q_weights_granularity)
            == len(q_activations_granularity)
        )

        qconfig_dict['quantization']['op_wise'] = {}

        for w_bit, a_bit, w_mode, a_mode, w_granularity, a_granularity, module_name in zip(
            q_weights_bit,
            q_activations_bit,
            q_weights_mode,
            q_activations_mode,
            q_weights_granularity,
            q_activations_granularity,
            regex_module_names,
        ):
            w_bit, a_bit = _convert_dtype(w_bit, a_bit)
            qconfig = qconfig_parse('minmax', w_bit, w_mode, w_granularity, 'kl', a_bit, a_mode, a_granularity)
            qconfig_dict['quantization']['op_wise'][module_name] = qconfig

        return qconfig_dict

    else:  # global quantization
        w_bit, a_bit = q_weights_bit, q_activations_bit
        w_mode, a_mode = q_weights_mode, q_activations_mode
        w_granularity, a_granularity = q_weights_granularity, q_activations_granularity

        w_bit, a_bit = _convert_dtype(w_bit, a_bit)

        qconfig = qconfig_parse('minmax', w_bit, w_mode, w_granularity, 'kl', a_bit, a_mode, a_granularity)

        qconfig_dict['quantization']['model_wise'] = qconfig
        return qconfig_dict


def inc_quantize(
    model_fp: torch.nn.Module,
    qconfig_dict: dict,
    data_loader: torch.utils.data.DataLoader = None,
    mp_calibration_samples: int = None,
) -> neural_compressor.model.torch_model.PyTorchFXModel:
    '''
    Parameters:
        model_fp: float point model
        qconfig_dict: inc qconfig_dict
        data_loader: torch.utils.data.DataLoader
        mp_calibration_samples: number of samples for calibration
    Return:
        model_qt: quantized model
    '''
    model_fp.eval()

    if mp_calibration_samples is not None:
        qconfig_dict['quantization']['calibration']['sampling_size'] = mp_calibration_samples

    # # ============== Quantization =============
    time_stamp = time.time()
    temp_yaml_name = f'temp_{time_stamp}.yaml'
    with open(temp_yaml_name, 'w') as f:
        yaml.dump(qconfig_dict, f)
    try:
        quantizer = Quantization(temp_yaml_name)
        quantizer.model = copy.deepcopy(model_fp)
        quantizer.calib_dataloader = data_loader
        model_qt = quantizer.fit()

        os.remove(temp_yaml_name)
    except Exception as e:
        os.remove(temp_yaml_name)
        raise Exception(e)

    return model_qt
