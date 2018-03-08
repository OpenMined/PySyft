sudo apt-get update
sudo apt-get install libssl-dev  -y

curl -O https://storage.googleapis.com/golang/go1.9.1.linux-amd64.tar.gz
tar -xvf go1.9.1.linux-amd64.tar.gz
rm go1.9.1.linux-amd64.tar.gz
sudo mv go /usr/local

echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.profile
export PATH=$PATH:/usr/local/go/bin

echo "export GOPATH=$HOME" >> ~/.profile
export GOPATH=$HOME

if [ ! -d "~/.ipfs" ]; then
  go get -u github.com/ipfs/ipfs-update
  ipfs-update install latest
  ipfs init
fi

# sudo apt-get install python3-dev # didn't seem to help
sudo apt-get install build-essential automake pkg-config libtool libffi-dev libgmp-dev -y
pip install -r requirements.txt

curl https://raw.githubusercontent.com/OpenMined/BootstrapNodes/master/bootstrap_nodes --output bootstrap_nodes
cat bootstrap_nodes | xargs ipfs bootstrap add

ipfs daemon --enable-pubsub-experiment  > ipfs.log 2> ipfs.err &

nohup ./bin/start_worker &
