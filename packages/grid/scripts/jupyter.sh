#!/bin/bash

# $1 is the PySyft dir
# $2 is jupyter token

# this runs jupyter notebooks on port 8888
pidof -o %PPID -x $0 >/dev/null && echo "ERROR: Script $0 already running" && exit 1

cd $1
tox -e syft.jupyter $2
