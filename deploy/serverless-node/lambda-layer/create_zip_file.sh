#!/bin/bash

#Build a zip file containing all dependencies of PyGrid Node, to deploy to an AWS Lambda Layer.
#The root file should be called `Python`, and contains all the dependencies.

mkdir python
python3.8 -m pip install -r requirements.txt -t python

# Note: We have used python3.8 specifically because the python
# version used to install the dependencies in the lambda layer must
# be the same as the one used to install the syft dependency on the EFS.

#Let us remove the unnecessary files now, and create a zip file to upload to lambda layer.
cd python
rm -r *.whl *.dist-info __pycache__    # Remove unnecessary files
cd ..
zip -r pygrid-node-dep.zip python

# Remove unnecessary folders
rm -rf python