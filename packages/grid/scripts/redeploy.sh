#!/bin/bash

# only run one redeploy.sh at a time
pidof -o %PPID -x $0 >/dev/null && echo "ERROR: Script $0 already running" && exit 1

# cronjob logs: $ tail -f /var/log/syslog | grep -i cron

# $1 is the PySyft dir
# $2 is the git repo like: https://github.com/OpenMined/PySyft
# $3 is the branch like: dev
# $4 is the permission user like: om
# $5 is the permission group like: om
# $6 is the node type like: domain
# $7 is the node name like: node
# $8 is the build directory where we copy the source so we dont trigger hot reloading

echo "Code has changed so redeploying with HAGrid"
rm -rf ${8}
cp -r ${1} ${8}
chown -R ${4}:${5} ${8}
/usr/sbin/runuser -l ${4} -c "pip install -e ${1}/packages/hagrid"
# /usr/sbin/runuser -l ${4} -c "hagrid launch ${7} ${6} to localhost --repo=${2} --branch=${3} --ansible_extras='docker_volume_destroy=true'"
/usr/sbin/runuser -l ${4} -c "hagrid launch ${7} ${6} to localhost --repo=${2} --branch=${3}"
