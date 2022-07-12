import datetime
import subprocess

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()


def get_git_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('UTF-8').strip()


def get_build_name():
    # TODO (Maciej): WW should be based on the commit date, not current date
    return 'ww{}.{}-{}'.format(
        datetime.datetime.utcnow().strftime("%V"),
        datetime.datetime.utcnow().isoweekday(),
        get_git_hash()
    )


def get_dependencies():
    deps = []
    with open('requirements.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if 'tensorflow' not in line.lower() and 'tf-' not in line.lower():
                deps.append(line)
    return deps


setup(
    name='dynast',
    version=get_build_name(),
    description='',  # TODO(Maciej) Add description
    long_description='',  # TODO(Maciej) Add long description
    long_description_content_type="text/markdown",
    author='Cummings, Daniel J; Munoz, Pablo; Nittur Sridhar, Sharath; Sarah, Anthony; Sundaresan, '
           'Sairam; Szankin, Maciej; Webb, Tristan;',
    author_email='daniel.j.cummings@intel.com; pablo.munoz@intel.com; sharath.nittur.sridhar@intel.com; '
                 'anthony.sarah@intel.com; sairam.sundaresan@intel.com; maciej.szankin@intel.com; '
                 'tristan.webb@intel.com',
    license='Intel Confidential',
    packages=find_packages(),
    install_requires=get_dependencies(),
    zip_safe=False,
    entry_points={
            'console_scripts': [
                'dynast=dynast.cli:main',
            ],
    },
)
