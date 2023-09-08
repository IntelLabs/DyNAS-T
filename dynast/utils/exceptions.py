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


from dynast.supernetwork.supernetwork_registry import SUPERNET_METRICS, SUPERNET_PARAMETERS


class InvalidSupernetException(Exception):
    def __init__(self, supernet):
        self.message = f'Invalid super-network ({supernet}) specified. Choose from the following: {list(SUPERNET_PARAMETERS.keys())}'
        super().__init__(self.message)


class InvalidMetricsException(Exception):
    def __init__(self, supernet, metric):
        valid_metrics = SUPERNET_METRICS[supernet]
        self.message = (
            f'Invalid metric specified: {metric}. Super-network f{supernet} supports following metrics: {valid_metrics}'
        )
        super().__init__(self.message)
