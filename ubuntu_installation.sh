sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install python3-pip python3-dev -y
sudo apt-get install libssl-dev  -y
sudo pip3 install -r requirements.txt

sudo curl -O https://storage.googleapis.com/golang/go1.9.1.linux-amd64.tar.gz
sudo tar -xvf go1.9.1.linux-amd64.tar.gz
sudo mv go /usr/local

echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.profile
export PATH=$PATH:/usr/local/go/bin

echo "export GOPATH=$HOME" >> ~/.profile
export GOPATH=$HOME

go get -u github.com/ipfs/ipfs-update
ipfs-update install latest
ipfs init
ipfs daemon --enable-pubsub-experiment  > ipfs.log &


