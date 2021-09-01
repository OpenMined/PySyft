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

# these commands cant be used because they trigger hot reloading
# however without them accidental changes to the working tree might cause issues
# with the fetch process so we should consider changing how this works perhaps by
# copying the code into a folder for execution and keeping the git repo seperate

# git checkout main --force
# git branch -D $3 || true
# git checkout $3 --force

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
    git reset --hard
    git checkout $3 --force
    git pull origin $3 --rebase
    chown -R $4:$5 .
fi

if [ "$CURRENT_BRANCH" != "$3" ]
then
    echo "Checking out branch: ${3}"
fi

git reset --hard
git checkout $3 --force
git pull origin $3 --rebase
chown -R $4:$5 .

END_HASH=$(git rev-parse HEAD)
CONTAINER_HASH=$(docker exec $(docker ps --format "{{.Names}}" | grep frontend) env | grep VERSION_HASH | sed 's/VERSION_HASH=//')

SCRIPT_PATH="$(dirname \"$0\")"

if [ "$START_HASH" != "$END_HASH" ]
then
    echo "Git hashes dont match, redeploying"
    bash /home/om/PySyft/packages/grid/scripts/redeploy.sh $1 $2 $3 $4 $5 $6 $7 $8
elif [[ ! "$END_HASH" == *"$CONTAINER_HASH"* ]]
then
    echo "Container hash doesnt match code, redeploying"
    bash /home/om/PySyft/packages/grid/scripts/redeploy.sh $1 $2 $3 $4 $5 $6 $7 $8
fi
echo "Finished autoupdate CRON"
