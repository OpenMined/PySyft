<h1 align="center">

  <br>
  <a href="http://duet.openmined.org/"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/logo_big.png" alt="PySyft" width="200"></a>
  <br>
  A library for computing on data<br /> you do not own and cannot see
  <br>

</h1>

<div align="center"> <a href="https://pypi.org/project/syft/"><img src="https://pepy.tech/badge/syft" /></a> <a href="https://pypi.org/project/syft/"><img src="https://badge.fury.io/py/syft.svg" /></a> <a href="https://github.com/OpenMined/PySyft/actions/workflows/syft-version_tests.yml"><img src="https://github.com/OpenMined/PySyft/actions/workflows/syft-version_tests.yml/badge.svg?branch=dev" /></a> <a href="https://openmined.github.io/PySyft/dev/bench/"><img src="https://github.com/OpenMined/PySyft/actions/workflows/syft-benchmark.yml/badge.svg?branch=dev" /></a><br /> <a href="#"><img src="https://github.com/OpenMined/PySyft/workflows/Tutorials/badge.svg" /></a> <a href="https://openmined.slack.com/messages/support"><img src="https://img.shields.io/badge/chat-on%20slack-7A5979.svg" /></a> <a href="https://mybinder.org/v2/gh/OpenMined/PySyft/main"><img src="https://mybinder.org/badge.svg" /></a> <a href="http://colab.research.google.com/github/OpenMined/PySyft/blob/main"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
<br /><br />

<div align="center"><a href="#"><img src="https://stars.medv.io/openmined/pysyft.svg" /></a></div>

</div>

# PySyft is a Python library for secure and private Deep Learning.

PySyft decouples private data from model training, using
[Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html),
[Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy),
and Encrypted Computation (like
[Multi-Party Computation (MPC)](https://en.wikipedia.org/wiki/Secure_multi-party_computation)
and [Homomorphic Encryption (HE)](https://en.wikipedia.org/wiki/Homomorphic_encryption))
within the main Deep Learning frameworks like PyTorch and TensorFlow. Join the movement on
[Slack](http://slack.openmined.org/).

---

Most software libraries let you compute over the information you own and see inside of machines you control. However, this means that you cannot compute on information without first obtaining (at least partial) ownership of that information. It also means that you cannot compute using machines without first obtaining control over those machines. This is very limiting to human collaboration and systematically drives the centralization of data, because you cannot work with a bunch of data without first putting it all in one (central) place.

The Syft ecosystem seeks to change this system, allowing you to write software which can compute over information you do not own on machines you do not have (total) control over. This not only includes servers in the cloud, but also personal desktops, laptops, mobile phones, websites, and edge devices. Wherever your data wants to live in your ownership, the Syft ecosystem exists to help keep it there while allowing it to be used privately for computation.

## 0.5.0 Release

The current stable release is `0.5.0` which is available on:

- [PyPI](https://pypi.org/project/syft/)
- [Docker Hub](https://hub.docker.com/u/openmined)

For many use cases you can simply use:

```
$ pip install -U syft --pre
```

If you are developing against this version please use the [`syft_0.5.0`](https://github.com/OpenMined/pysyft/tree/syft_0.5.0) branch.

## Examples

The examples inside the `/packages/syft/examples/` folder are currently compatible with `0.5.0` however we will be updating and moving examples down to the root `/notebooks` folder in the `dev` branch as we begin work on `0.6.0`.

## Mono Repo 🚝

This repo contains multiple projects which work together, namely PySyft and PyGrid.

```
OpenMined/PySyft
├── README.md
└── packages
    ├── grid    <-- Code formerly @ OpenMined/PyGrid
    └── syft    <-- You are here 📌
```

_NOTE_ Changing the entire folder structure will likely result in some minor issues.
If you spot one please let us know or open a PR.

## PySyft

PySyft is the centerpiece of the Syft ecosystem. It has two primary purposes. You can either use PySyft to perform two types of computation:

1. _Dynamic:_ Directly compute over data you cannot see.
2. _Static:_ Create static graphs of computation which can be deployed/scaled at a later date on different compute.

The [PyGrid library](https://github.com/OpenMined/PyGrid) serves as an API for the management and deployment of PySyft at scale. It also allows for you to extend PySyft for the purposes of Federated Learning on web, mobile, and edge devices using the following Syft worker libraries:

- [KotlinSyft](https://github.com/OpenMined/KotlinSyft) (Android)
- [SwiftSyft](https://github.com/OpenMined/SwiftSyft) (iOS)
- [syft.js](https://github.com/OpenMined/syft.js) (Javascript)
- PySyft (Python, you can use PySyft itself as one of these "FL worker libraries")

However, the Syft ecosystem only focuses on consistent object serialization/deserialization, core abstractions, and algorithm design/execution across these languages. These libraries alone will not connect you with data in the real world. The Syft ecosystem is supported by the Grid ecosystem, which focuses on the deployment, scalability, and other additional concerns around running real-world systems to compute over and process data (such as data compliance web applications).

- PySyft is the library that defines objects, abstractions, and algorithms.
- [PyGrid](https://github.com/OpenMined/PyGrid) is the platform which lets you deploy them within a real institution.
- [PyGrid Admin](https://github.com/OpenMined/pygrid-admin) is a UI which allows a data owner to manage their PyGrid deployment.

A more detailed explanation of PySyft can be found in the
[white paper on Arxiv](https://arxiv.org/abs/1811.04017).

PySyft has also been explained in videos on YouTube:

- [PriCon Sep 2020 Duet Demo](https://www.youtube.com/watch?v=DppXfA6C8L8&ab_channel=OpenMined)
- [Introduction to Privacy-Preserving AI using PySyft by @iamtrask](https://www.youtube.com/watch?v=NJBBE_SN90A)

## Pre-Installation

PySyft is available on PyPI and Conda.

We recommend that you install PySyft within a virtual environment like
[Conda](https://docs.anaconda.com/anaconda/user-guide/getting-started/),
due to its ease of use. If you are using Windows, we suggest installing
[Anaconda and using the Anaconda
Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/) to
work from the command line.

```bash
$ conda create -n pysyft python=3.9
$ conda activate pysyft
$ conda install jupyter notebook
```

## Version Support

We support **Linux**, **MacOS** and **Windows** and the following Python and Torch versions.
Older versions may work, however we have stopped testing and supporting them.

| Py / Torch | 1.6 | 1.7 | 1.8+ |
| ---------- | --- | --- | ---- |
| 3.8        | ✅  | ✅  | ✅   |
| 3.9        | ➖  | ✅  | ✅   |
| 3.10       | ➖  | ✅  | ✅   |

## Installation

### Pip

```bash
$ pip install -U syft --pre
```

This will auto-install PyTorch and other dependencies as required to run the
examples and tutorials. For more information on building from source see the contribution guide [here](https://github.com/OpenMined/PySyft/blob/main/packages/syft/CONTRIBUTING.md).

## Documentation

Coming soon! Until then, please view the Examples below.

## Examples

A comprehensive list of examples can be found [here](https://github.com/OpenMined/PySyft/tree/main/packages/syft/examples).

These tutorials cover a variety of Python libraries for data science and machine learning.

All the examples can be played with by launching a Jupyter Notebook and navigating to the `examples` folder.

```bash
$ jupyter notebook
```

### Duet

<a href="https://github.com/OpenMined/PySyft/tree/main/packages/syft/examples/duet"><img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/logo_duet.png" alt="PySyft" width="350"></a>

Duet is a peer-to-peer tool within PySyft that provides a research-friendly API for a Data Owner to privately expose their data, while a Data Scientist can access or manipulate the data on the owner's side through a zero-knowledge access control mechanism. It's designed to lower the barrier between research and privacy-preserving mechanisms, so that scientific progress can be made on data that is currently inaccessible or tightly controlled. **The main benefit of using Duet is that allows you to get started using PySyft, without needing to manage a full PyGrid deployment. It is the simplest path to using Syft, without needing to install anything (except Syft 😉).**

You can find all [Duet examples](https://github.com/OpenMined/PySyft/tree/main/packages/syft/examples/duet) in the `examples/duet` folder.

## Contributing

The guide for contributors can be found [here](https://github.com/OpenMined/PySyft/blob/main/packages/syft/CONTRIBUTING.md).
It covers all that you need to know to start contributing code to PySyft today.

Also, join the rapidly growing community of 12,000+ on [Slack](http://slack.openmined.org).
The Slack community is very friendly and great about quickly answering questions about the use and development of PySyft!

## Disclaimer

This software is in beta. Use at your own risk.

## A quick note about 0.2.x

The PySyft 0.2.x codebase is now in its own branch [here](https://github.com/OpenMined/PySyft/tree/syft_0.2.x), but OpenMined will not offer official support for this version range. We have compiled a list of [FAQs](https://github.com/OpenMined/PySyft/blob/main/packages/syft/docs/FAQ_0.2.x.md) relating to this version.\_

## Support

For support in using this library, please join the **#support** Slack channel. [Click here to join our Slack community!](https://slack.openmined.org)

## Organizational Contributions

We are very grateful for contributions to PySyft from the following organizations!

- [<img src="https://github.com/udacity/private-ai/blob/master/udacity-logo-vert-white.png?raw=true" alt="Udacity" width="160"/>](https://udacity.com/)
- [<img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/organizations/comind.png" alt="coMind" width="160" />](https://github.com/coMindOrg/federated-averaging-tutorials)
- [<img src="https://i.ibb.co/vYwcG9N/arkhn-logo.png" alt="Arkhn" width="160" />](http://ark.hn)
- [<img src="https://raw.githubusercontent.com/dropoutlabs/files/master/dropout-labs-logo-white-2500.png" alt="Dropout Labs" width="160"/>](https://dropoutlabs.com/)
- [<img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/organizations/genbu.png" alt="GENBU AI" width="160"/>](https://genbu.ai/)
- [<img src="https://raw.githubusercontent.com/OpenMined/PySyft/main/packages/syft/docs/img/organizations/bitdefender.png" alt="Bitdefender" width="160"/>](https://www.bitdefender.com/) |

## License

[Apache License 2.0](https://github.com/OpenMined/PySyft/blob/main/packages/syft/LICENSE)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FOpenMined%2FPySyft.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FOpenMined%2FPySyft?ref=badge_large)
