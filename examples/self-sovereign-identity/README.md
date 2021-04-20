# SSI Authentication for Duet Demo

In this demo we use Hyperledger Aries agents to establish DIDComm connections between the Scientist and the Data Owner. The data owner then authenticates the scientist by requesting proof of a credential. The scientist must first be issued this credential by a test OM Issuer.

This demo uses docker and docker-compose to simplify management of the Hyperledger dependencies.

## Requirements

You will need:

- Docker: https://www.docker.com/products/docker-desktop
- Bash compatible Shell (or WSL on Windows)
- The source-to-image (s2i) tool
  - MacOS
    `$ brew install source-to-image`
  - Linux
    Download, extract and add s2i to your PATH:
    https://github.com/openshift/source-to-image/releases
  - Windows
    Download, extract and add s2i to your PATH:
    https://github.com/openshift/source-to-image/releases

If you have issues with s2i please consult the `s2i Detailed Instructions` below.

## Setup

Ensure that Docker is running. If it is not try `sudo dockerd` in another terminal
In one terminal spin up `ACA-Py agents` for the DataOwner, DataScientist and DuetIssuer

- Setup a [PySyft environment](https://github.com/OpenMined/PySyft/blob/dev/docs/installing.rst)
- Install the aries-basic-controller in this environment `$ pip install aries-basic-controller`
- In the PySyft environment open up Jupyter Notebooks - `$ jupyter notebook` - Navigate to /examples/self-sovereign-identity/notebooks

- In a separate terminal instance:
  From the /examples/self-sovereign-identity directory: - `$ ./manage start`
- Follow the notebooks

## s2i Detailed Instructions

s2i is required to build the docker images used in the demo and can be downloaded [here](https://github.com/openshift/source-to-image/releases).
The website gives instructions for installing on other platforms. Verify that s2i is in your PATH. If not, then edit your PATH and add the directory where s2i is installed. The manage script will look for the s2i executable on your PATH. If it is not found you will get a message asking you to download and set it on your PATH.

      - If you are using a Mac and have Homebrew installed, the following command will install s2i: `$ brew install source-to-image`
      - If you are using Linux, go to the releases page and download the correct distribution for your machine. Choose either the linux-386 or the linux-amd64 links for 32 and 64-bit, respectively. Unpack the downloaded tar with tar -xvf "Release.tar.gz"
      - If you are using windows you will need to choose the windows binary on the github releases page.
      - You should now see an executable called s2i. Either add the location of s2i to your PATH environment variable, or move it to a pre-existing directory in your PATH. For example, `sudo cp /path/to/s2i /usr/local/bin` will work with most setups. You can test it using s2i version. For Windows consult Google for how to edit your PATH or use Windows Subsystem for Linux so that you can use the Linux instructions above.
