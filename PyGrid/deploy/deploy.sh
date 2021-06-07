#!/bin/bash

## For debugging
# redirect stdout/stderr to a file
exec &> log.out


echo 'Simple Web Server for testing the deployment'
sudo apt update -y
sudo apt install apache2 -y
sudo systemctl start apache2
echo """
<h1 style='color:#f09764; text-align:center'>
    OpenMined First Server Deployed via Terraform
</h1>
""" | sudo tee /var/www/html/index.html

echo 'Setup Miniconda environment'

sudo wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
sudo bash miniconda.sh -b -p miniconda
sudo rm miniconda.sh
export PATH=/miniconda/bin:$PATH > ~/.bashrc
conda init bash
source ~/.bashrc
conda create -y -n pygrid python=3.7
conda activate pygrid

echo 'Install poetry...'
pip install poetry

echo 'Install GCC'
sudo apt-get install python3-dev -y
sudo apt-get install libevent-dev -y
sudo apt-get install gcc -y

echo 'Cloning PyGrid'
git clone https://github.com/OpenMined/PyGrid


while IFS=, read -r id port
do
    echo "Start PyGrid $id on port $port"

    if [[ "$id" == *"network"* ]]; then
        echo "Running $id"
        cd /PyGrid/apps/network
        poetry install
        nohup ./run.sh --port $port --start_local_db

    elif [[ "$id" == *"node"* ]]; then
        echo "Running $id"
        cd /PyGrid/apps/node
        poetry install
        ./run.sh --id $id --port $port --start_local_db

    elif [[ "$id" == *"worker"* ]]; then
        echo "Starting Worker"
        cd /PyGrid/apps/worker
        poetry install
    else
        echo "Only Network, Nodes, & Workers"
    fi
done < /home/ubuntu/pygrid.txt
