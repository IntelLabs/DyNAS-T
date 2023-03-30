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


# # Once for All: Train One Network and Specialize it for Efficient Deployment
# # Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# # International Conference on Learning Representations (ICLR), 2020.

import os
import sys

import numpy as np
import torch

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = [
    "get_same_padding",
    "sub_filter_start_end",
    "min_divisible_value",
    "val2list",
    "download_url",
    "write_log",
    "DistributedTensor",
]


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def download_url(url, model_dir="~/.torch/", overwrite=False):
    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
        return None


def write_log(logs_path, log_str, prefix="valid", should_print=True, mode="a"):
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)
    """ prefix: valid, train, test """
    if prefix in ["valid", "test"]:
        with open(os.path.join(logs_path, "valid_console.txt"), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if prefix in ["valid", "test", "train"]:
        with open(os.path.join(logs_path, "train_console.txt"), mode) as fout:
            if prefix in ["valid", "test"]:
                fout.write("=" * 10)
            fout.write(log_str + "\n")
            fout.flush()
    else:
        with open(os.path.join(logs_path, "%s.txt" % prefix), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if should_print:
        print(log_str)


class DistributedTensor(object):
    # TODO(macsz) This most likely can be dropped (unless there is a strong case for Horovod/distributed)
    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = torch.zeros(1)[0]
        self.synced = False

    def update(self, val, delta_n=1):
        val *= delta_n
        if self.sum is None:
            self.sum = val.detach()
        else:
            self.sum += val.detach()
        self.count += delta_n

    @property
    def avg(self):
        import horovod.torch as hvd

        if not self.synced:
            self.sum = hvd.allreduce(self.sum, name=self.name)
            self.synced = True
        return self.sum / self.count
