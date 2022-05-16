import functools as _functools
import json
import logging as _logging
import os
import subprocess
import time as _time

import pandas as pd
from rich.console import Console as _Console
from rich.logging import RichHandler as _RichHandler

console = _Console()
_logging.basicConfig(
    level='INFO',
    format="%(message)s",
    datefmt='[%X]',
    handlers=[_RichHandler()],
)

log = _logging.getLogger('rich')


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
