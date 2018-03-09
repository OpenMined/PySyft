sudo apt-get update
sudo apt-get install curl libssl-dev  -y

if [ ! -d "$HOME/.ipfs" ]; then
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
  ipfs init
fi

# sudo apt-get install python3-dev # didn't seem to help
sudo apt-get install build-essential automake pkg-config libtool libffi-dev libgmp-dev -y
pip install -r requirements.txt
