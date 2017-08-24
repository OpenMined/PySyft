# Syft

[![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://openmined.slack.com/)

> Homomorphically Encrypted Deep Learning Library

The goal of this library is to give the user the ability to efficiently train Deep Learning models in a homomorphically encrypted state without needing to be an expert in either. Furthermore, by understanding the characteristics of both Deep Learning and Homomorphic Encryption, we hope to find very performant combinations of the two.  See [notebooks](./notebooks) folder for tutorials on how to use the library.

- [Setup](#setup-install-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Start](#start)
  - [Tests](#tests)
- [For Contributors](#for-contributors)
- [Relevant Literature](#relevant-literature)
- [License](#license)

## Setup

### Prerequisites

You need to install this library locally before running any of the notebooks this repository or the [main demonstration](https://github.com/OpenMined/sonar):

```sh
# Get dependencies ready
pip install -r requirements.txt
# install the lib locally
python setup.py install
```

### Installation

#### For Anaconda Users

```
bash install_for_anaconda_users.sh
```

##### Windows

```sh
conda install -c conda-forge gmpy2
pip install -r requirements.txt
python setup.py install
```

#### For Docker Users

Install Docker from [its website](https://www.docker.com/).
For macOS users with [Homebrew](https://brew.sh/) installed, use `brew cask install docker`

## Usage

### Start

Then, run:

```sh
git clone https://github.com/OpenMined/PySyft.git
cd PySyft
make run
```

### Tests

```sh
cd PySyft
make test
```

## For Contributors

If you are interested in contributing to Syft, first check out our [Contributor Quickstart Guide](https://github.com/OpenMined/Docs/blob/master/contributing/quickstart.md) and then sign into our [Slack Team](https://openmined.slack.com/) channel #syft to let us know which projects sound interesting to you! (or propose your own!).

## Relevant Literature

As both Homomorphic Encryption and Deep Learning are still somewhat sparsely known, below is a curated list of relevant reading materials to bring you up to speed with the major concepts and themes of these exciting fields.

### Encrypted Deep Learning - Recommended Reading:
- [How to build a fully encrypted AI model (trained on unencrypted data)](http://iamtrask.github.io/2017/03/17/safe-ai/)
- [Simple secure protocol for federated machine learning (using a python-paillier library)](https://blog.n1analytics.com/distributed-machine-learning-and-partially-homomorphic-encryption-1/)
- [Prototype for using encrypted AI to preserve user privacy (in python)](http://iamtrask.github.io/2017/06/05/homomorphic-surveillance/)
- [Manual for Using Homomorphic Encryption for Bioinformatics (paper)](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/11/ManualHE-3.pdf)

### Homomorphic Encryption - Recommended Reading:
- [A Comparison of the Homomorphic Encryption Schemes](https://eprint.iacr.org/2014/062.pdf)
- [Homomorphic Encryption API Software Library](http://heat-h2020-project.blogspot.co.uk/2017/02/homomorphic-encryption-api-software.html)
- [Faster Homomorphic Function Evaluation using Non-Integral Base Encoding](http://heat-h2020-project.blogspot.co.uk/2017/)

### Relevant Papers:
- [Encrypted accelerated least squares regression](http://proceedings.mlr.press/v54/esperanca17a/esperanca17a.pdf)
- [PSML 2017 - Abstracts](https://sites.google.com/view/psml/program/abstracts)
- [Encrypted statistical machine learning: new privacy
preserving methods](https://arxiv.org/pdf/1508.06845.pdf)
- [A review of homomorphic encryption and software
tools for encrypted statistical machine learning](https://arxiv.org/pdf/1508.06574.pdf)
- [Privacy-Preserving Distributed Linear Regression on High-Dimensional Data](https://eprint.iacr.org/2016/892)
- [Secure Computation With Fixed-Point Numbers](https://www1.cs.fau.de/filepool/publications/octavian_securescm/secfp-fc10.pdf)

### Related Libraries:
- [HomomorphicEncryption - An R package for fully homomorphic encryption](http://www.louisaslett.com/HomomorphicEncryption/#details)
- [A Secure Multiparty Computation (MPC) protocol for computing linear regression on vertically distributed datasets](https://github.com/iamtrask/linreg-mpc)
- [Dask Tutorial](https://github.com/dask/dask-tutorial)

### Related Blogs:
- [Private Deep Learning with MPC - A Simple Tutorial from Scratch](https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/)
- [Secret Sharing, Part 1 - Distributing Trust and Work](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/)
- [Secret Sharing, Part 2 - Efficient Sharing with the Fast Fourier Transform](https://mortendahl.github.io/2017/06/24/secret-sharing-part2/)
- [Distributed machine learning and partially homomorphic encryption (Part 1)](https://blog.n1analytics.com/distributed-machine-learning-and-partially-homomorphic-encryption-1/)
- [Tutorial: How to verify crowdsourced training data using a Known Answer Review Policy](https://blog.mturk.com/tutorial-how-to-verify-crowdsourced-training-data-using-a-known-answer-review-policy-85596fb55ed)

## License

[Apache-2.0](https://github.com/OpenMined/PySyft/blob/develop/LICENSE) by OpenMined contributors
