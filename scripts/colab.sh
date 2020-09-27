#!/bin/bash
# set -e

# update the code (handy during development)
echo "> git pull"
git pull --rebase 2>&1 >/dev/null

# show the latest commit
echo "> using latest commit"
# make it line up with the indent above
echo "  `  git log --oneline | sed -n 1p`"

# step 2 install PySyft
echo "> installing pip requirements"
pip install -e . 2>&1 >/dev/null

# step 3 install PySyft Dependencies
pip install -r requirements.txt 2>&1 >/dev/null

# step 4 patch linux python 3.6.9 on Google Colab
# without this fix the code doesnt work in some places, weirdly this code is not broken
# on my MacOS Python 3.6.9

# first we show the line with the missing getattr(obj, attr, None)
# cat /usr/local/lib/python3.6/dist-packages/typeguard/__init__.py | grep "'__origin__')"

# then we backup the file
# cp /usr/local/lib/python3.6/dist-packages/typeguard/__init__.py /usr/local/lib/python3.6/dist-packages/typeguard/__init__.py.bak

# then we patch it
echo "> patching python 3.6.9 colab bug"
sed -i "s/'__origin__')/'__origin__', None)/" /usr/local/lib/python3.6/dist-packages/typeguard/__init__.py

# then we show its changed
# cat /usr/local/lib/python3.6/dist-packages/typeguard/__init__.py | grep "'__origin__')"

# now you can see its the same as the others
# cat /usr/local/lib/python3.6/dist-packages/typeguard/__init__.py | grep "'__origin__', None)"
echo "> colab environment ready"