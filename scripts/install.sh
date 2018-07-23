#!/bin/sh
pip install requirements.txt
conda install pytorch=0.3.1 -c soumith
python setup.py install