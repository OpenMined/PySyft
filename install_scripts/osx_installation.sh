#
# Check if Homebrew is installed
#
which -s brew
if [[ $? != 0 ]] ; then
    # Install Homebrew
    # https://github.com/mxcl/homebrew/wiki/installation
    /usr/bin/ruby -e "$(curl -fsSL https://raw.github.com/gist/323731)"
fi

#
# Check if Git, go, Python3, and python dependencies are installed
#
which -s go || brew install go
which -s wget || brew install wget
which -s git || brew install git
which -s python3 || brew install python3

if [ ! -d "$HOME/.ipfs" ]; then
  wget https://dist.ipfs.io/ipfs-update/v1.5.2/ipfs-update_v1.5.2_darwin-amd64.tar.gz --no-check-certificate
  tar -xvf ipfs-update_v1.5.2_darwin-amd64.tar.gz
  cd ipfs-update
  sh install.sh
  cd ../
  rm -rf ipfs-update*

  ipfs-update install latest

  ipfs init
fi

pip3 install http://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl 
pip3 install torchvision
