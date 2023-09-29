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


from dynast.supernetwork.supernetwork_registry import (
    EVALUATION_INTERFACE,
    SUPERNET_METRICS,
    get_all_supported_metrics,
    get_supported_supernets,
)


def test_get_supported_supernets():
    assert get_supported_supernets() == list(EVALUATION_INTERFACE.keys())


def test_get_all_supported_metrics():
    metrics = []
    for s, ms in SUPERNET_METRICS.items():
        metrics.extend(ms)
    unique_metrics = list(set(metrics))
    assert unique_metrics == get_all_supported_metrics()
