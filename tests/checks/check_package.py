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

import os

import pytest

from tests.checks.helpers import get_packages_paths

ROOT_DIR = os.getcwd()


@pytest.mark.parametrize('dir_path', get_packages_paths(ROOT_DIR))
def test_check_init_files(dir_path):
    fn = os.path.join(dir_path, '__init__.py')
    assert os.path.exists(fn), f"Package {dir_path} does not contain `__init__.py` file!"
