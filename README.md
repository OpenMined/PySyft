# Introduction

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/OpenMined/PySyft/master) [![Build Status](https://travis-ci.org/OpenMined/PySyft.svg?branch=torch_1)](https://travis-ci.org/OpenMined/PySyft) [![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://openmined.slack.com/messages/team_pysyft) [![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft.svg?type=small)](https://app.fossa.io/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft?ref=badge_small)

PySyft is a Python library for secure, private Deep Learning. PySyft decouples private data from model training, using [Multi-Party Computation (MPC)](https://en.wikipedia.org/wiki/Secure_multi-party_computation) within PyTorch. Join the movement on [Slack](http://slack.openmined.org/).

## PySyft in Detail

A more detailed explanation of PySyft can be found in the [paper on arxiv](https://arxiv.org/abs/1811.04017)

PySyft has also been explained in video form by [Siraj Raval](https://www.youtube.com/watch?v=39hNjnhY7cY&feature=youtu.be&a=)

## Installation

> PySyft supports Python >= 3.6 and PyTorch 1.1.0

```bash
pip install syft
```

You can also install PySyft from source on a variety of operating systems by following this [installation guide](https://github.com/OpenMined/PySyft/blob/dev/INSTALLATION.md).

## Run Local Notebook Server

All the examples can be played with by running the command

```bash
make notebook
```

and selecting the pysyft kernel

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

[<img src="https://github.com/udacity/private-ai/blob/master/udacity-logo-vert-white.png?raw=true" alt="Udacity" width="200"/>](https://udacity.com/) | [<img src="https://raw.githubusercontent.com/coMindOrg/federated-averaging-tutorials/master/images/comindorg_logo.png" alt="coMind" width="200" height="130"/>](https://github.com/coMindOrg/federated-averaging-tutorials)
-------------------------------------------------------------- | ---------------------------------------------------------------------------

## Disclaimer

Do NOT use this code to protect data (private or otherwise) - at present it is very insecure.

## License

[Apache License 2.0](https://github.com/OpenMined/PySyft/blob/master/LICENSE)

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft?ref=badge_large)
