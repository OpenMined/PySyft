# Introduction

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/OpenMined/PySyft/master) [![Build Status](https://travis-ci.org/OpenMined/PySyft.svg?branch=torch_1)](https://travis-ci.org/OpenMined/PySyft) [![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://openmined.slack.com/messages/team_pysyft) [![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft.svg?type=small)](https://app.fossa.io/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft?ref=badge_small)

PySyft is a Python library for secure, private Deep Learning. PySyft decouples private data from model training, using [Multi-Party Computation (MPC)](https://en.wikipedia.org/wiki/Secure_multi-party_computation) within PyTorch. Join the movement on [Slack](http://slack.openmined.org/).

## PySyft in Detail

A more detailed explanation of PySyft can be found in the [paper on arxiv](https://arxiv.org/abs/1811.04017)

PySyft has also been explained in video form by [Siraj Raval](https://www.youtube.com/watch?v=39hNjnhY7cY&feature=youtu.be&a=)

## Pre-Installation

Optionally, we recommend that you install PySyft within the [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/overview.html) virtual environment. If you are using Windows, I suggest installing [Anaconda and using the Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/) to work from the command line.

```bash
conda create -n pysyft python=3
conda activate pysyft # some older version of conda require "source activate pysyft" instead.
conda install jupyter notebook
```

## Installation

> PySyft supports Python >= 3.6 and PyTorch 1.1.0

```bash
pip install syft
```

If you have an installation error regarding zstd, run this command and then re-try installing syft.

```bash
pip install --upgrade --force-reinstall zstd
```
If this still doesn't work, and you happen to be on OSX, make sure you have [OSX command line tools](https://railsapps.github.io/xcode-command-line-tools.html) installed and try again.

You can also install PySyft from source on a variety of operating systems by following this [installation guide](https://github.com/OpenMined/PySyft/blob/dev/INSTALLATION.md).

## Run Local Notebook Server

All the examples can be played with by running the command

```bash
make notebook
```

and selecting the pysyft kernel

## Use the Docker image

Instead of installing all the dependencies on your computer, you can run a notebook server (which comes with Pysyft installed) using [Docker](https://www.docker.com/). All you will have to do is start the container like this:

```bash
$ docker container run youben/pysyft:miniconda
[I 23:41:51.429 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 23:41:52.865 NotebookApp] Serving notebooks from local directory: /workspace
[I 23:41:52.865 NotebookApp] The Jupyter Notebook is running at:
[I 23:41:52.865 NotebookApp] http://172.17.0.2:8888/?token=e0bb2ed0c185ae35b02f512ff293ed97fa0d2300b2ff73b3
[I 23:41:52.866 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 23:41:52.870 NotebookApp] No web browser found: could not locate runnable browser.
[C 23:41:52.870 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-10-open.html
    Or copy and paste one of these URLs:
        http://172.17.0.2:8888/?token=e0bb2ed0c185ae35b02f512ff293ed97fa0d2300b2ff73b3

```

You can use the provided link to access the jupyter notebook (the link is only accessible from your local machine).


You could also build the image on your own and run it locally:

```bash
$ cd docker-image
$ docker image build -t pysyft-notebook .
Sending build context to Docker daemon    129kB
Step 1/11 : FROM continuumio/miniconda3
 ---> 6b5cf97566c3
Step 2/11 : ENV WORKSPACE /workspace
 ---> Using cache
 ---> 81438bdb049e
Step 3/11 : RUN apt-get update && apt-get install -y gcc
 ---> Using cache
 ---> c853d3634a16
Step 4/11 : RUN conda install jupyter notebook
 ---> Using cache
 ---> 8d06f83cefb9
Step 5/11 : RUN pip install --no-cache-dir syft numpy
 ---> Using cache
 ---> b11a980f4474
Step 6/11 : RUN mkdir $WORKSPACE
 ---> Using cache
 ---> 225f5364c24a
Step 7/11 : WORKDIR $WORKSPACE
 ---> Using cache
 ---> 8e5ca393c858
Step 8/11 : RUN git clone https://github.com/udacity/private-ai
 ---> Using cache
 ---> 30c282e7797b
Step 9/11 : COPY ./entrypoint.sh /
 ---> Using cache
 ---> 8ccc772fd4b9
Step 10/11 : RUN chmod +x /entrypoint.sh
 ---> Using cache
 ---> 17d339b42fde
Step 11/11 : ENTRYPOINT ["/entrypoint.sh"]
 ---> Using cache
 ---> 7c400194576f
Successfully built 7c400194576f
Successfully tagged pysyft-notebook:latest
$
$ docker container run youben/pysyft:miniconda
[I 23:41:51.429 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 23:41:52.865 NotebookApp] Serving notebooks from local directory: /workspace
[I 23:41:52.865 NotebookApp] The Jupyter Notebook is running at:
[I 23:41:52.865 NotebookApp] http://172.17.0.2:8888/?token=e0bb2ed0c185ae35b02f512ff293ed97fa0d2300b2ff73b3
[I 23:41:52.866 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 23:41:52.870 NotebookApp] No web browser found: could not locate runnable browser.
[C 23:41:52.870 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-10-open.html
    Or copy and paste one of these URLs:
        http://172.17.0.2:8888/?token=e0bb2ed0c185ae35b02f512ff293ed97fa0d2300b2ff73b3
```

You shouldn't see the same output as this during the build process, since the image was already built in this machine, but it should only take a moment to build  (depending on your internet connection).

## Try out the Tutorials

A comprehensive list of tutorials can be found [here](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials)

These tutorials cover how to perform techniques such as federated learning and differential privacy using PySyft.

## Start Contributing

The guide for contributors can be found [here](https://github.com/OpenMined/PySyft/tree/master/CONTRIBUTING.md). It covers all that you need to know to start contributing code to PySyft in an easy way.

Also join the rapidly growing community of 3700+ on [Slack](http://slack.openmined.org). The slack community is very friendly and great about quickly answering questions about the use and development of PySyft!

## Troubleshooting

We have written an installation example in [this colab notebook](https://colab.research.google.com/drive/14tNU98OKPsP55Y3IgFtXPfd4frqbkrxK), you can use it as is to start working with PySyft on the colab cloud, or use this setup to fix your installation locally.

## Organizational Contributions

We are very grateful for contributions to PySyft from the following organizations!

[<img src="https://github.com/udacity/private-ai/blob/master/udacity-logo-vert-white.png?raw=true" alt="Udacity" width="200"/>](https://udacity.com/) | [<img src="https://raw.githubusercontent.com/coMindOrg/federated-averaging-tutorials/master/images/comindorg_logo.png" alt="coMind" width="200" height="130"/>](https://github.com/coMindOrg/federated-averaging-tutorials) | [<img src="https://arkhn.org/img/arkhn_logo_black.svg" alt="Arkhn" width="200" height="130"/>](http://ark.hn) | [<img src="https://raw.githubusercontent.com/dropoutlabs/files/master/dropout-labs-logo-white-2500.png" alt="Dropout Labs" width="200"/>](https://dropoutlabs.com/)
--------------------------------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------

## Disclaimer

Do NOT use this code to protect data (private or otherwise) - at present it is very insecure. Come back in a couple months.

## License

[Apache License 2.0](https://github.com/OpenMined/PySyft/blob/master/LICENSE)

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft?ref=badge_large)
