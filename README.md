# Introduction

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/OpenMined/PySyft/master) [![Build Status](https://travis-ci.org/OpenMined/PySyft.svg?branch=master)](https://travis-ci.org/OpenMined/PySyft) [![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://openmined.slack.com/messages/team_pysyft) [![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft.svg?type=small)](https://app.fossa.io/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft?ref=badge_small)

PySyft is a Python library for secure, private Deep Learning. PySyft decouples private data from model training, using [Multi-Party Computation \(MPC\)](https://en.wikipedia.org/wiki/Secure_multi-party_computation) within PyTorch. Join the movement on [Slack](http://slack.openmined.org/).

## See PySyft in Action

* [Emulate remote PyTorch execution](https://colab.research.google.com/drive/1vsgH0ydHyel5VRAxO2yhRQfXYUuIYkp5) - This notebook demonstrates the tensor passing between workers, though both the workers live in the same environment.
* Emulate remote PyTorch execution using sockets: [Server](https://colab.research.google.com/drive/1-Jb_E_nDuBGHIJ_psI95k-ukh-P_aly-#scrollTo=lrcghOJOWGHw) \| [Client](https://colab.research.google.com/drive/1Je1rk7olA9uTWWaqvvt4_gXf7yX1rTBm) - This notebook demonstrates the tensor passing and remote execution, with workers living in different environments.

  > Note: Run Server before Client

* [Federated Learning](https://colab.research.google.com/drive/1F3ALlA3ogfeeVXuwQwVoX4PimzTDJhPy#scrollTo=PTCvX6H9JDCt) - This notebook demonstrates the model training over distributed data \(data belonging to multiple owners\).

## Docker

```bash
git clone https://github.com/OpenMined/PySyft.git
cd PySyft
scripts/run_docker.sh
```

> Image size: 644MB

The container mount the examples folder on a volume so every change on the notebooks is persistent. Furthermore the container is deleted when it is stopped, in a way to facilitate development. You just have to change PySyft code, and run the run\_docker.sh script to observe changes you've made on notebooks.

## Installation

> PySyft supports Python &gt;= 3.6 and PyTorch 0.3.1

Pick the proper PyTorch version according to your machine: [CPU](http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl) \| [CUDA9.1](http://download.pytorch.org/whl/cu91/torch-0.3.1-cp36-cp36m-linux_x86_64.whl) \| [CUDA9.0](http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl) \| [CUDA8.0](http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl)

```bash
conda install pytorch=0.3.1 -c soumith
pip3 install -r requirements.txt
python3 setup.py install
```

On Windows use the following steps to install PyTorch 0.3.1-

```text
conda install -c peterjc123 pytorch
conda install -c peterjc123 pytorch-cpu
```

## Run Unit Tests

```text
python3 setup.py test
```
Alternatively, tests can be run with
```bash
make test
```
## Run Local Notebook Server
All the examples can be played with by running the command
```bash
make notebook
```
and selecting the pysyft kernel

Join the rapidly growing community of 2500+ on [Slack](http://slack.openmined.org) and help us in our mission. We are really friendly people!

## Organizational Contributions

We are very grateful for contributions to PySyft from the following organizations!

 ![drawing](https://raw.githubusercontent.com/coMindOrg/federated-averaging-tutorials/master/images/comindorg_logo.png)  
  
 [coMind Website](https://comind.org/) & [coMind Github](https://github.com/coMindOrg/federated-averaging-tutorials)

## Disclaimer

Do NOT use this code to protect data (private or otherwise) - at present it is very insecure. 

## License

[Apache License 2.0](https://github.com/OpenMined/PySyft/blob/master/LICENSE)

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft?ref=badge_large)

