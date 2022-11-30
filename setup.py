import datetime
import subprocess

from setuptools import find_packages, setup


def get_git_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('UTF-8').strip()


def get_build_name():
    # TODO (Maciej): WW should be based on the commit date, not current date
    return 'ww{}.{}-{}'.format(
        datetime.datetime.utcnow().strftime("%V"), datetime.datetime.utcnow().isoweekday(), get_git_hash()
    )


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
    version=get_build_name(),
    description='DyNAS-T (Dynamic Neural Architecture Search Toolkit) - a SuperNet NAS optimization package',
    long_description='DyNAS-T (Dynamic Neural Architecture Search Toolkit) is a SuperNet NAS optimization package '
    'designed for finding the optimal Pareto front during neural architure search while minimizing '
    'the number of search validation measurements. It supports single-/multi-/many-objective '
    'problems for a variety of domains supported by the Intel AI Lab HANDI framework. The system '
    'currently heavily utilizes the pymoo optimization library.',
    long_description_content_type="text/markdown",
    author='Cummings, Daniel J; Nittur Sridhar, Sharath; Sarah, Anthony; Sundaresan, Sairam; ' 'Szankin, Maciej;',
    author_email='daniel.j.cummings@intel.com; sharath.nittur.sridhar@intel.com; anthony.sarah@intel.com; '
    'sairam.sundaresan@intel.com; maciej.szankin@intel.com;',
    license='Intel Confidential',  # TODO(Maciej) Update license
    packages=find_packages(),
    install_requires=get_dependencies(),
    extras_require={
        'test': get_test_dependencies(),
    },
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'dynast=dynast.cli:main',
        ],
    },
)
