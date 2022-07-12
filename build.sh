#!/bin/bash

rm -Rf build/ dist/

python setup.py sdist bdist_wheel
