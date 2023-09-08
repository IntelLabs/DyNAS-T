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
# import datetime

import os
import re

from setuptools import find_packages, setup


def get_version():
    try:
        filepath = "./dynast/version.py"
        with open(filepath) as version_file:
            (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
    except Exception as error:
        assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)
    return __version__


def _read_requirements(fn):
    deps = []
    with open(fn) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            deps.append(line)
    return deps


def get_dependencies(feature=None):
    deps = []

    if feature is None:
        fns = ['requirements.txt']
    elif feature == 'all':
        fns = [f for f in os.listdir('.') if f.startswith('requirements') and f.endswith('.txt')]
    else:
        fns = ['requirements.txt', f'requirements_{feature}.txt']

    deps = []

    for fn in fns:
        deps.extend(_read_requirements(fn))

    # TODO(macsz) At the moment, PyPi will produce an error if you attempt to install NNCF from GitHub. This is because
    # the necessary features have not yet been included in any released versions of NNCF, so it must be removed from
    # this list and installed manually by the user.
    # For now, to install NNCF, use the following steps:
    # ```
    # git clone https://github.com/openvinotoolkit/nncf.git
    # cd nncf
    # git checkout e0bc50359992a3d73d4ed3e6396c8b4f1d4ae631
    # pip install -e .
    # ```
    # Once version 2.6 of NNCF is released, the following line can be deleted:
    deps = [d for d in deps if 'nncf' not in d]

    deps = list(sorted(set(deps)))
    return deps


setup(
    name='dynast',
    version=get_version(),
    description='DyNAS-T (Dynamic Neural Architecture Search Toolkit) - a SuperNet NAS optimization package',
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author='Maciej Szankin, Sharath Nittur Sridhar, Anthony Sarah, Sairam Sundaresan',
    author_email='maciej.szankin@intel.com, sharath.nittur.sridhar@intel.com, '
    'anthony.sarah@intel.com, sairam.sundaresan@intel.com',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=get_dependencies(),
    extras_require={
        'all': get_dependencies('all'),
        'test': get_dependencies('test'),
        'bootstrapnas': get_dependencies('bootstrapnas'),
        'neural_compressor': get_dependencies('neural_compressor'),
    },
    entry_points={
        'console_scripts': [
            'dynast=dynast.cli:main',
        ],
    },
    python_requires='>=3.7.0',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
