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
from glob import glob

IGNORE_DIRS = [
    '.tox',
    'dist',
    '.git',
    'htmlcov',
    'log_dir',
    '.idea',
    '.venv',
]


def get_python_files(root_dir, exclude_files=None):
    if not exclude_files:
        exclude_files = []

    files = []
    code_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(d) and d not in IGNORE_DIRS]
    pattern = "*.py"

    for code_dir in code_dirs:
        for d, _, _ in os.walk(code_dir):
            files.extend(glob(os.path.join(d, pattern)))

    return list(sorted(set(files) - set(exclude_files)))
