#!/bin/bash
# $1 is the PySyft dir
# $2 is the git repo like: https://github.com/OpenMined/PySyft
# $3 is the branch like: dev
# $4 is the permission user like: om
# $4 is the permission group like: om
cd $1
git fetch $2 $3 && git checkout $3 --force
chown -R $4:$5 .