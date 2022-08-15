import functools as _functools
import json
import logging
import os
import subprocess
import time as _time

import pandas as pd
import requests


def set_logger(level: int = logging.INFO):
    """Create logger object and set the logging level to `level`."""
    global log
    log = logging.getLogger()
    log.setLevel(level)
    log.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(filename)s:%(lineno)d - %(message)s",
        "%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    # Disable PIL polluting logs with it's debug logs: https://github.com/camptocamp/pytest-odoo/issues/15
    logging.getLogger('PIL').setLevel(logging.WARNING)


log = None
set_logger()


def measure_time(func):
    """ Decorator to measure elapsed time of a function call.

    Usage:

    ```
    @measure_time
    def foo(bar=2):
        return [i for i in range(bar)]

    print(foo())
    # Will print:
    # > Calling foo
    # [0, 1, 2]
    # > Finished foo in 0.0004 s
    ```
    """
    @_functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        log.info("> Calling {}".format(func.__name__))
        start_time = _time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = _time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        log.info('> Finished {} in {:.4f} s'.format(func.__name__, run_time))
        return value
    return wrapper_timer


def samples_to_batch_multiply(base_samples, batch_size):
    return (base_samples//batch_size+1)*batch_size


def get_hostname():
    return os.getenv('HOSTNAME', os.getenv('HOST', 'unnamed-host'))


def get_cores(num_cores: int):
    """ For a given number of cores, returns the core IDs that should be used.

    This script prioritizes using cores from the same socket first. e.g. for a
    two socket CLX 8280 system, that means using cores: 0-27, 56-83, 28-55, 84-111
    in that order, since [0-27, 56-83] belong to the same socket.
    """
    cmd = ['lscpu', '--json', '--extended']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, code = p.communicate()
    cpu = json.loads(out)

    df = pd.DataFrame.from_dict(cpu['cpus'])
    for key in ['cpu', 'node', 'socket', 'core']:
        df[key] = df[key].astype(int)

    df = df.sort_values(['node', 'socket', 'cpu'])
    cores = df['cpu'].to_list()[:num_cores]
    cores = [str(c) for c in cores]
    return ','.join(cores)


def get_remote_file(remote_url: str, model_dir: str, overwrite: bool = False) -> str:
    """Download remote file and save it locally. Returns full path to saved file.

    Note: If used to download OFA supernets set `model_dir` to `.torch/ofa_nets`. It's a default
    path that OFA uses and it's hardcoded.

    Arguments:
    ----------
    * remote_url: String with remote file's URL.
    * model_dir: Directory to which the file will be saved.
    * overwrite: Overwrite existing file if set and file exists .

    Returns:
    --------
    * Absolute path to saved file
    """

    if not os.path.isdir(model_dir) or not os.path.exists(model_dir):
        log.error(f"{model_dir} is not a directory")
        raise NotADirectoryError(f"{model_dir} is not a directory")

    fname = os.path.basename(remote_url)
    save_path = os.path.join(os.path.abspath(model_dir), fname)

    if os.path.exists(save_path) and not overwrite:
        log.info(f'File {save_path} exists, skipping download of {remote_url}')
        return save_path

    log.info(f'Downloading {remote_url} to {save_path}')
    data = requests.get(remote_url)

    try:
        data.raise_for_status()
    except requests.exceptions.HTTPError as he:
        log.error(f'Fetching {remote_url} failed. Return code: {data.status_code}')
        raise he

    with open(save_path, 'wb') as f:
        f.write(data.content)

    return save_path
