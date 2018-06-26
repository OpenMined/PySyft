# PySyft 

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/OpenMined/PySyft/master)
[![Build Status](https://travis-ci.org/OpenMined/PySyft.svg?branch=master)](https://travis-ci.org/OpenMined/PySyft)

PySyft is a Python library for secure, private Deep Learning. PySyft decouples private data from model training, using [Multi-Party Computation (MPC)](https://en.wikipedia.org/wiki/Secure_multi-party_computation) over PyTorch and tensorflow.  
Join the movement on [Slack](http://slack.openmined.org/).

## See PySyft in Action
- [Emulate remote PyTorch execution](https://colab.research.google.com/drive/1vsgH0ydHyel5VRAxO2yhRQfXYUuIYkp5) - This notebook demonstrates the tensor passing between workers, though both the workers live in the same environment.
- Emulate remote PyTorch execution using sockets: [Server](https://colab.research.google.com/drive/1-Jb_E_nDuBGHIJ_psI95k-ukh-P_aly-#scrollTo=lrcghOJOWGHw) | [Client](https://colab.research.google.com/drive/1Je1rk7olA9uTWWaqvvt4_gXf7yX1rTBm) - This notebook demonstrates the tensor passing and remote execution, with workers living in different environments.
  > Note: Run Server before Client
- [Federated Learning](https://colab.research.google.com/drive/1F3ALlA3ogfeeVXuwQwVoX4PimzTDJhPy#scrollTo=PTCvX6H9JDCt) - This notebook demonstrates the model training over distributed data (data belonging to multiple owners).

## Installation

```
pip install -r requirements.txt

python3 setup.py install
```

## Run Unit Tests

```
python3 setup.py test
```
