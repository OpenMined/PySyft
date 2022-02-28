#!/bin/bash

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

# these commands cant be used because they trigger hot reloading
# however without them accidental changes to the working tree might cause issues
# with the fetch process so we should consider changing how this works perhaps by
# copying the code into a folder for execution and keeping the git repo seperate

# git checkout main --force
# git branch -D $3 || true
# git checkout $3 --force

pidof -o %PPID -x $0 >/dev/null && echo "ERROR: Script $0 already running" && exit 1

cd $1
START_HASH=$(git rev-parse HEAD)
CURRENT_REMOTE=$(git remote -v | head -n 1 | cut -d ' ' -f 1 | awk '{print $2}')
CURRENT_BRANCH=$(git branch --show-current)
echo "Running autoupdate CRON"

# does https://github.com/OpenMined/PySyft contain OpenMined/PySyft
if [[ ! "$CURRENT_REMOTE" == *"$2"* ]]
then
    echo "Switching remotes to: ${2}"
    git remote rm origin || true
    git remote add origin https://github.com/$2
    git fetch origin
    echo "Checking out branch: ${3}"
    git reset --hard "origin/${3}"
    git checkout "origin/${3}" --force
    chown -R $4:$5 .
fi

if [ "$CURRENT_BRANCH" != "$3" ]
then
    echo "Checking out branch: ${3}"
fi

git fetch origin
git reset --hard "origin/${3}"
git checkout "origin/${3}" --force
chown -R $4:$5 .

END_HASH=$(git rev-parse HEAD)
CONTAINER_HASH=$(docker ps --format "{{.Names}}" | grep 'backend' | head -1l | xargs -I {} docker exec {} env | grep VERSION_HASH | sed 's/VERSION_HASH=//')

REDEPLOY="0"

# see hagrid --release options
if [[ ${11} = "development" ]]; then
    RELEASE=development
else
    RELEASE=production
fi

if [[ "$START_HASH" != "$END_HASH" ]]
then
    echo "Git hashes $START_HASH vs $END_HASH dont match, redeploying"
    REDEPLOY="1"
elif [[ ! "$END_HASH" == *"$CONTAINER_HASH"* ]]
then
    echo "Container hash $END_HASH not in $CONTAINER_HASH, redeploying"
    REDEPLOY="1"
elif [[ -z "$CONTAINER_HASH" ]]
then
    echo "Container hash $CONTAINER_HASH is not valid, redeploying"
    REDEPLOY="1"
fi

echo "START_HASH=$START_HASH"
echo "END_HASH=$END_HASH"
echo "CONTAINER_HASH=$CONTAINER_HASH"
echo "REDEPLOY=$REDEPLOY"

if [[ ${REDEPLOY} != "0" ]]; then
    bash /home/om/PySyft/packages/grid/scripts/redeploy.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${RELEASE}
fi

echo "Finished autoupdate CRON"
