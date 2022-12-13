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

import json
import os
import re
import subprocess

import numpy as np
from openvino.inference_engine import IECore

from dynast.quantization.policy import OVQuantizationPolicy
from dynast.utils import get_cores, get_hostname, log, measure_time
from dynast.utils.onnx import save_onnx_model

if 'OPENVINO_DIR' not in os.environ:
    OPENVINO_DIR = '/opt/intel/openvino_2021/deployment_tools/'
else:
    OPENVINO_DIR = os.environ['OPENVINO_DIR']

OV_BENCHMARK_PATH = os.path.join(OPENVINO_DIR, 'tools/benchmark_tool/benchmark_app.py')
OV_MODEL_OPTIMIZER_PATH = os.path.join(OPENVINO_DIR, 'model_optimizer/mo.py')


@measure_time
def load_openvino(
    folder: str,
    name: str,
    return_net: bool = False,
    is_quantized=False,
):
    ie = IECore()
    if is_quantized:
        folder = os.path.join(folder, 'optimized')
    net = ie.read_network(model=os.path.join(folder, name + '.xml'), weights=os.path.join(folder, name + '.bin'))

    exec_net = ie.load_network(net, 'CPU')
    if return_net:
        return exec_net, net
    else:
        return exec_net


@measure_time
def save_ov_quantized(
    tmp_folder: str = '/store/.torch/',
    model_name: str = 'tmp',
    quant_policy: str = 'DefaultQuantization',
    stat_subset_size: int = 3 * 128,
) -> None:

    fp32_path_xml = os.path.join(tmp_folder, model_name + '.xml')
    fp32_path_bin = os.path.join(tmp_folder, model_name + '.bin')
    q_path = os.path.join(tmp_folder, '')

    policy_to_use = OVQuantizationPolicy.get_policy(
        model_name=model_name,
        fp32_path_xml=fp32_path_xml,
        fp32_path_bin=fp32_path_bin,
        quant_policy=quant_policy,
        stat_subset_size=stat_subset_size,
    )

    json_fname = os.path.join(tmp_folder, 'cfg_tmp.json')

    with open(json_fname, 'w') as f:
        json.dump(policy_to_use, f, indent=4)

    cmd_pot = [
        'python',
        '/opt/intel/openvino_2021/deployment_tools/tools/post_training_optimization_toolkit/main.py',
        '-c',
        json_fname,
        '--output-dir',
        q_path,
        '-d',
    ]
    log.info('Running external command: {}'.format(' '.join(cmd_pot)))
    result = subprocess.run(cmd_pot)
    assert result.returncode == 0, cmd_pot


@measure_time
def save_openvino(model, shape, tmp_folder='/store/.torch/', name='tmp', verbose=False):
    os.makedirs(tmp_folder, exist_ok=True)
    onnx_path = os.path.join(tmp_folder, name + '.onnx')

    shape_onnx = np.array(shape)

    with open(onnx_path, 'wb') as f:
        save_onnx_model(model, f, tuple(shape_onnx))

    # Path to OpenVino subgraph extensions. For example '/store/code/ofa/ofa/ov_extensions/'
    OFA_OV_EXT_PATH = os.environ['OFA_OV_EXT_PATH']

    # convert from ONNX to openvino IR
    cmd = [
        'python',
        OV_MODEL_OPTIMIZER_PATH,
        '--input_model',
        onnx_path,
        '--input_shape',
        str(list(shape)),
        '--output_dir',
        tmp_folder,
        '--model_name',
        name,
        '--extensions',
        OFA_OV_EXT_PATH,
    ]

    if verbose is False:
        cmd += ['--silent']

    log.info('Running external command: {}'.format(' '.join(cmd)))
    result = subprocess.run(cmd)
    assert result.returncode == 0


@measure_time
def benchmark_openvino(
    shape, experiment_name=None, perf_folder=None, time=20, cores=None, nstreams=1, is_quantized=False
):
    # NOTE(Maciej) Model has to already be quentized and saved.
    if not experiment_name:
        experiment_name = '{}_dynast_eval'.format(get_hostname())

    if type(cores) == int:
        log.info(
            'Param `cores` is of type `int`. Converting to a list of cores: {} -> {}'.format(cores, get_cores(cores))
        )
        cores = get_cores(cores)

    folder_name = os.path.expanduser('/store/.torch/{}'.format(experiment_name))
    ov_model_dir = os.path.join(folder_name, 'ov_model')

    if is_quantized:
        file_path = os.path.join(ov_model_dir, 'optimized', experiment_name + '.xml')
    else:
        file_path = os.path.join(ov_model_dir, experiment_name + '.xml')

    if cores is not None:
        cmd = ['taskset', '-c', cores]
    else:
        cmd = []

    cmd += ['python', OV_BENCHMARK_PATH, '-m', file_path]
    cmd += ['-d', 'CPU', '-b', str(shape[0]), '-t', str(time), '-nstreams', str(nstreams), '-api', 'sync']

    if perf_folder is not None:
        cmd += ['-report_type', 'detailed_counters', '-report_folder', perf_folder]

    log.info('Running external command: {}'.format(' '.join(cmd)))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, code = p.communicate()

    if code == 0:
        log.error('FAILED')
        log.error(cmd)
    # parse output
    numbers = re.findall('\d*\.?\d+', str(out))
    latency_ov = float(numbers[-2])
    fps_ov = float(numbers[-1])

    return latency_ov, fps_ov
