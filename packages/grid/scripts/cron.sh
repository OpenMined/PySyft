#!/bin/bash
# $1 is the PySyft dir
# $2 is the git repo like: https://github.com/OpenMined/PySyft
# $3 is the branch like: dev
# $4 is the permission user like: om
# $5 is the permission group like: om
# $6 is the node type like: domain
# #7 is the node name like: node
cd $1
START_HASH=$(git rev-parse HEAD)
# git checkout main --force
# git branch -D $3 || true
git remote rm origin || true
git remote add origin https://github.com/$2
git fetch origin
git checkout $3 --force
git pull origin $3 --rebase
chown -R $4:$5 .
END_HASH=$(git rev-parse HEAD)

if [ "$START_HASH" != "$END_HASH" ]; then
    echo "Code has changed to running hagrid"
    ANSIBLE_CONFIG=/home/om/PySyft/packages/grid/ansible.cfg ansible-playbook -i 127.0.0.1, /home/om/PySyft/packages/grid/ansible/site.yml -e "node_type=$6 node_name=$7 github_repo=$2 repo_branch=$3 deploy_only=true"
fi
