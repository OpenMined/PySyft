#!/bin/bash
echo 'Updating'
sudo apt-get update -y

echo 'Installing python3.8'
sudo apt-get install python3.8 -y
sudo apt-get install python3-pip -y

echo 'Moving to python3.8'
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --auto python3

echo 'Install python3.8-dev'
sudo apt-get install python3.8-dev -y

echo 'Install AV dependencies'
sudo apt-get install libsrtp2-dev -y
sudo apt-get install libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev libavfilter-dev libavdevice-dev -y

echo 'Updating pip'
pip3 install -U pip

echo 'Installing nfs'
sudo apt-get install nfs-common -y

echo 'Mount EFS to /efs'
mkdir /efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport $EFS_DNS:/ /efs

echo 'Give persmission to write to /efs'
sudo chmod go+rw /efs

echo 'Installing syft in ~/efs/dep'
mkdir -p ~/efs/dep
python3 -m pip install syft==0.2.9 --target ~/efs/dep

echo 'Replacing torch-gpu with torch-cpu'
python3 -m pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --target efs/dep --upgrade