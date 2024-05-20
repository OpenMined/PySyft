#!/bin/bash
rm -rf build dist
python setup.py bdist_wheel
rm -rf build
