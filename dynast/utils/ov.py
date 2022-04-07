import json
import os
import re
import subprocess

import numpy as np
from openvino.inference_engine import IECore

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
):
    ie = IECore()

    net = ie.read_network(
        model=os.path.join(folder, name + '.xml'),
        weights=os.path.join(folder, name + '.bin')
    )

    exec_net = ie.load_network(net, 'CPU')
    if return_net:
        return exec_net, net
    else:
        return exec_net


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
    cmd = ['python', OV_MODEL_OPTIMIZER_PATH,
           '--input_model', onnx_path, '--input_shape', str(list(shape)),
           '--output_dir', tmp_folder, '--model_name', name, '--extensions', OFA_OV_EXT_PATH]

    if verbose is False:
        cmd += ['--silent']

    log.info('Running external command: {}'.format(' '.join(cmd)))
    result = subprocess.run(cmd)
    assert result.returncode == 0


@measure_time
def benchmark_openvino(shape, experiment_name=None, perf_folder=None, time=20, cores=None, nstreams=1):
    # NOTE(Maciej) Model has to already be quentized and saved.
    if not experiment_name:
        experiment_name = '{}_dynast_eval'.format(get_hostname())

    if type(cores) == int:
        log.info('Param `cores` is of type `int`. Converting to a list of cores: {} -> {}'.format(cores, get_cores(cores)))
        cores = get_cores(cores)

    folder_name = os.path.expanduser('/store/.torch/{}'.format(experiment_name))
    ov_model_dir = os.path.join(folder_name, 'ov_model')

    file_path = os.path.join(ov_model_dir, experiment_name + '.xml')

    if cores is not None:
        cmd = ['taskset', '-c', cores]
    else:
        cmd = []

    cmd += ['python', OV_BENCHMARK_PATH, '-m', file_path]
    cmd += ['-d', 'CPU',
            '-b', str(shape[0]),
            '-t', str(time),
            '-nstreams', str(nstreams),
            '-api', 'sync'
            ]

    if perf_folder is not None:
        cmd += ['-report_type', 'detailed_counters', '-report_folder',
                perf_folder]

    log.info('Running external command: {}'.format(' '.join(cmd)))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, code = p.communicate()

    if (code == 0):
        print('FAILED')
        print(cmd)
    # parse output
    numbers = re.findall('\d*\.?\d+', str(out))
    latency_ov = float(numbers[-2])
    fps_ov = float(numbers[-1])

    return latency_ov, fps_ov
