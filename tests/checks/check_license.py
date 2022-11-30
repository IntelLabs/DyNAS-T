# INTEL CONFIDENTIAL
# Copyright 2022 Intel Corporation. All rights reserved.

# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the
# express license under which they were provided to you ("License"). Unless the License provides otherwise, you may
# not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without
# Intel's prior written permission.

# This software and the related documents are provided as is, with no express or implied warranties, other than those
# that are expressly stated in the License.

# This software is subject to the terms and conditions entered into between the parties.

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
