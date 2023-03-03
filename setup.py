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

from setuptools import find_packages, setup


def get_version():
    # TODO(macsz) Replace with __version__
    return '1.1.0rc3'


def get_dependencies():
    deps = []
    with open('requirements.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            deps.append(line)
    return deps


def get_test_dependencies():
    deps = []
    with open('requirements_test.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            deps.append(line)
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
        'test': get_test_dependencies(),
    },
    entry_points={
        'console_scripts': [
            'dynast=dynast.cli:main',
        ],
    },
    python_requires='>=3.7.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
    ],
)
