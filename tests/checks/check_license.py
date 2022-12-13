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

from tests.checks.helpers import get_python_files

ROOT_DIR = os.getcwd()


def get_license_template():
    """Retrive license from LICENSE.md to be used as a template to compare against."""

    license_path = os.path.join(ROOT_DIR, 'LICENSE.md')
    license_content = ''
    with open(license_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                license_content += line + ' '
    return license_content


def get_license_header(path):
    license_content = ''
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    # NOTE(macsz): Downside of this approach is that all license header lines have to start with
                    # a '#' comment sign.
                    line = line.replace('#', '').strip()
                    if line:
                        license_content += line + ' '
                else:
                    break
    return license_content


license_target = get_license_template()


@pytest.mark.parametrize('fp', get_python_files(ROOT_DIR))
def test_check_license(fp):
    license_file = get_license_header(fp)
    assert license_target in license_file, f"Wrong license header in `{fp}`"
