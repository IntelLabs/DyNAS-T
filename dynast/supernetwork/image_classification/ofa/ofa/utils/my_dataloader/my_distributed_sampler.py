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


import math

import torch
from torch.utils.data.distributed import DistributedSampler

__all__ = [
    "MyDistributedSampler",
]


class MyDistributedSampler(DistributedSampler):
    """Allow Subset Sampler in Distributed Training"""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, sub_index_list=None):
        super(MyDistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        self.sub_index_list = sub_index_list  # numpy

        self.num_samples = int(math.ceil(len(self.sub_index_list) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        print("Use MyDistributedSampler: %d, %d" % (self.num_samples, self.total_size))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.sub_index_list), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        indices = self.sub_index_list[indices].tolist()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
