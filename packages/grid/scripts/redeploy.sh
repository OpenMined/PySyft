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
# $9 is a bool for enabling tls or not, where true is tls enabled
# $10 is the path to tls certs if available
# $11 release mode, production or development with hot reloading
# $12 docker_tag if set to local, normal local build occurs, otherwise use dockerhub

if [[ ${11} = "development" ]]; then
    RELEASE=development
else
    RELEASE=production
fi

if [[ -z "${12}" ]]; then
    DOCKER_TAG="local"
else
    DOCKER_TAG="${12}"
fi

echo "Code has changed so redeploying with HAGrid"
rm -rf ${8}
cp -r ${1} ${8}
chown -R ${4}:${5} ${8}
/usr/sbin/runuser -l ${4} -c "pip install -e ${8}/packages/hagrid"
# /usr/sbin/runuser -l ${4} -c "hagrid launch ${7} ${6} to localhost --repo=${2} --branch=${3} --ansible_extras='docker_volume_destroy=true'"
if [[ "${9}" = "true" ]]; then
    echo "Starting Grid with TLS"
    HAGRID_CMD="hagrid launch ${7} ${6} to localhost --repo=${2} --branch=${3} --release=${RELEASE} --tag=${DOCKER_TAG} --tls --cert_store_path=${10}"
    echo $HAGRID_CMD
    /usr/sbin/runuser -l ${4} -c "$HAGRID_CMD"
else
    echo "Starting Grid without TLS"
    HAGRID_CMD="hagrid launch ${7} ${6} to localhost --repo=${2} --branch=${3} --release=${RELEASE} --tag=${DOCKER_TAG}"
    echo $HAGRID_CMD
    /usr/sbin/runuser -l ${4} -c "$HAGRID_CMD"
fi
