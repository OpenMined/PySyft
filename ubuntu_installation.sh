sudo apt-get update
sudo apt-get install python3-pip python3-dev -y
sudo apt-get install libssl-dev  -y
sudo pip3 install -r requirements.txt


sudo apt-get install golang-1.9-go -y
echo "export GOPATH=/usr/lib/go-1.9
export PATH=$PATH:$GOROOT/bin:$GOPATH/bin" >> ~/.bashrc

source ~/.bashrc



