
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
rm Anaconda3-5.1.0-Linux-x86_64.sh
/home/ubuntu/anaconda3/bin/conda install pytorch-cpu torchvision -c pytorch

# pip install -r requirements.txt

sudo apt-get install libssl-dev  -y

curl -O https://storage.googleapis.com/golang/go1.9.1.linux-amd64.tar.gz
tar -xvf go1.9.1.linux-amd64.tar.gz
rm go1.9.1.linux-amd64.tar.gz
sudo mv go /usr/local

echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.profile
export PATH=$PATH:/usr/local/go/bin

echo "export GOPATH=$HOME" >> ~/.profile
export GOPATH=$HOME

go get -u github.com/ipfs/ipfs-update
ipfs-update install latest

if [ ! -d "~/.ipfs" ]; then
  ipfs init
fi

curl https://raw.githubusercontent.com/OpenMined/BootstrapNodes/master/bootstrap_nodes --output bootstrap_nodes
cat bootstrap_nodes | xargs ipfs bootstrap add

ipfs daemon --enable-pubsub-experiment  > ipfs.log 2> ipfs.err &

sudo add-apt-repository -y ppa:jonathonf/python-3.6

# sudo apt-get update
# sudo apt-get -y upgrade
# sudo apt-get install python3.6 -y
# sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
# sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2

# sudo apt-get install python3-pip python3.6-dev -y

# pip3 install -r requirements.txt