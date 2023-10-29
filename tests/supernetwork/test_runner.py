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

import pytest

from dynast.supernetwork.runner import Runner


def test_estimate_metric_invalid_predictor():
    runner = Runner(
        supernet='some_supernet',
        predictors={},
    )

    with pytest.raises(Exception):
        runner.estimate_metric(metric='some_metric', subnet_cfg={})
