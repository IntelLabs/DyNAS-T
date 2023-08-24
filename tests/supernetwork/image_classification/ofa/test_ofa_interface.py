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

from dynast.supernetwork.image_classification.ofa.ofa_interface import OFARunner


class TestOFARunner:
    def test_init_data_path_not_exist(self):
        with pytest.raises(FileNotFoundError):
            OFARunner(
                supernet='ofa_resnet50',
                dataset_path='/not-existing-path-to-a-dataset',
            )

    def test_init_data_path_none(self):
        OFARunner(
            supernet='ofa_resnet50',
            dataset_path=None,
        )
