#!/bin/sh

cd /workspace
jupyter notebook --ip=`cat /etc/hosts |tail -n 1|cut -f 1` --allow-root
