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
from glob import glob

IGNORE_DIRS = [
    '.tox',
    'dist',
    '.git',
    'htmlcov',
    'log_dir',
    '.idea',
    '.venv',
    'build',
    'nc_workspace',
    'docs',
    '__pycache__',
    '.pytest_cache',
    'dynast.egg-info',
    '.torch',
    '.github',
    'notebooks',
    'models',
    'out',
    'results',
    'tmp',
    'scripts',
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


def get_packages_paths(root_dir, exclude_files=None):
    if not exclude_files:
        exclude_files = []

    subdirs = []
    dirs = list(os.walk(root_dir))[1:]  # [1:] to exclude DyNAS-T's root dir
    for x in dirs:
        subdir = x[0]
        is_ok = True
        for ignored_dir in IGNORE_DIRS:
            if ignored_dir in subdir or f'./{ignored_dir}' in subdir:
                is_ok = False
        for ignored_dir in exclude_files:
            if ignored_dir in subdir or f'./{ignored_dir}' in subdir:
                is_ok = False

        # Check if subdir actually has any python files inside (or if there are subdirs)
        pys = [tmp_path for tmp_path in os.listdir(subdir) if tmp_path.endswith('.py') or os.path.isdir(tmp_path)]
        if not pys:
            continue

        if is_ok:
            subdirs.append(subdir)

    return subdirs
