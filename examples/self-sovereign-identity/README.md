# SSI Authentication for Duet Demo

In this demo we use Hyperledger Aries agents to establish DIDComm connections between the Scientist and the Data Owner. The data owner then authenticates the scientist by requesting proof of a credential. The scientist must first be issued this credential by a test OM Issuer.

This demo uses docker and docker-compose to simplify management of the Hyperledger dependencies.

## Requirements

You need to install:

1. Docker
2. The source-to-image (s2i) tool is also required to build the docker images used in the demo. S2I can be downloaded here. The website gives instructions for installing on other platforms. Verify that s2i is in your PATH. If not, then edit your PATH and add the directory where s2i is installed. The manage script will look for the s2i executable on your PATH. If it is not found you will get a message asking you to download and set it on your PATH.
      *  If you are using a Mac and have Homebrew installed, the following command will install s2i: brew install source-to-image
      *  If you are using Linux, go to the releases page and download the correct distribution for your machine. Choose either the linux-386 or the linux-amd64 links for 32 and 64-bit, respectively. Unpack the downloaded tar with tar -xvf "Release.tar.gz"
      *  If you are not sure about your Operating System you can visit this and/or follow the instructions.
      *  You should now see an executable called s2i. Either add the location of s2i to your PATH environment variable, or move it to a pre-existing directory in your PATH. For example, sudo cp /path/to/s2i /usr/local/bin will work with most setups. You can test it using s2i version.

Ensure that Docker is running. If it is not try sudo dockerd in another terminal.

## To start

In one terminal spin up ACA-Py agents for the DataOwner, DataScientist and DuetIssuer

* Run `./manage start`

* In another terminal initialise a [PySyft environment](https://github.com/OpenMined/PySyft/blob/dev/docs/installing.rst)

* Install the aries-basic-controller in this environment `pip install aries-basic-controller`

* Spin up the notebooks
    * cd notebooks
    * jupyter notebook