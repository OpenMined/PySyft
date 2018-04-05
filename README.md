# Syft

 [![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://openmined.slack.com/messages/team_pysyft)
[![Build Status](https://travis-ci.org/OpenMined/PySyft.svg?branch=master)](https://travis-ci.org/OpenMined/PySyft)
[![codecov](https://codecov.io/gh/openmined/pysyft/branch/master/graph/badge.svg)](https://codecov.io/gh/openmined/pysyft)
[![Help Contribute to Open Source](https://www.codetriage.com/openmined/pysyft/badges/users.svg)](https://www.codetriage.com/openmined/pysyft)
[![Backers on Open Collective](https://opencollective.com/OpenMined/backers/badge.svg)](#backers) 
[![Sponsors on Open Collective](https://opencollective.com/OpenMined/sponsors/badge.svg)](#sponsors)

> Homomorphically Encrypted Deep Learning Library

The goal of this library is to give the user the ability to efficiently train Deep Learning models in a homomorphically encrypted state, without needing to be an expert in either. Furthermore, by understanding the characteristics of both Deep Learning and Homomorphic Encryption, we hope to find a very performant combinations of the two. Also, check the [main demonstration](https://github.com/OpenMined/sonar) from the Sonar project.

- [Local setup](#local-setup)
- [For Contributors](#for-contributors)
- [Relevant Literature](#relevant-literature)
- [License](#license)

## Local setup

### Prerequisites

- Make sure Python 3.5+ in installed on your machine by checking `python3 --version`
- Set up a virtual environment for the Python libraries (optional, recommended)
- Install our OpenMined Unity app by following the guidelines [here](https://github.com/OpenMined/OpenMined)

### Python Requirements

The Python dependencies are listed in [`requirements.txt`](./requirements.txt) and can be installed through
```sh
pip3 install -r requirements.txt
```

#### PySyft installation
If you simply want to to _use_ PySyft, it is enough to install the library with:
```sh
python3 setup.py install
```
note: If you have anaconda installed, you can just run `bash install_for_anaconda.sh` for linux or `install_for_anaconda_on_windows.bat` for windows.

## For Contributors
If you are interested in contributing to Syft, first check out our [Contributor Quickstart Guide](https://github.com/OpenMined/Docs/blob/master/contributing/quickstart.md) and then sign into our [Slack Team](https://openmined.slack.com/) channel #team_pysyft to let us know which projects sound interesting to you! (or propose your own!).

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
- [Scalable and secure logistic regression via homomorphic encryption](https://pdfs.semanticscholar.org/d24c/81f1e2904ba6ec3f341161865ef93247855b.pdf)
- [ML Confidential: Machine Learning on Encrypted Data](https://eprint.iacr.org/2012/323.pdf)
- [CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf)
- [Privacy-Preserving Visual Learning Using Doubly Permuted Homomorphic Encryption](https://arxiv.org/pdf/1704.02203.pdf)

### Related Libraries:
- [HomomorphicEncryption - An R package for fully homomorphic encryption](http://www.louisaslett.com/HomomorphicEncryption/#details)
- [A Secure Multiparty Computation (MPC) protocol for computing linear regression on vertically distributed datasets](https://github.com/iamtrask/linreg-mpc)
- [Dask Tutorial](https://github.com/dask/dask-tutorial)
- [Charm-crypto](http://charm-crypto.io/)

### Related Blogs:
- [Private Deep Learning with MPC - A Simple Tutorial from Scratch](https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/)
- [Secret Sharing, Part 1 - Distributing Trust and Work](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/)
- [Secret Sharing, Part 2 - Efficient Sharing with the Fast Fourier Transform](https://mortendahl.github.io/2017/06/24/secret-sharing-part2/)
- [Distributed machine learning and partially homomorphic encryption (Part 1)](https://blog.n1analytics.com/distributed-machine-learning-and-partially-homomorphic-encryption-1/)
- [Distributed machine learning and partially homomorphic encryption (Part 2)](https://blog.n1analytics.com/distributed-machine-learning-and-partially-homomorphic-encryption-2/)
- [Tutorial: How to verify crowdsourced training data using a Known Answer Review Policy](https://blog.mturk.com/tutorial-how-to-verify-crowdsourced-training-data-using-a-known-answer-review-policy-85596fb55ed)

## Contributors

This project exists thanks to all the people who contribute. [[Contribute](CONTRIBUTING.md)].
<a href="graphs/contributors"><img src="https://opencollective.com/OpenMined/contributors.svg?width=890&button=false" /></a>


## Backers

Thank you to all our backers! 🙏 [[Become a backer](https://opencollective.com/OpenMined#backer)]

<a href="https://opencollective.com/OpenMined#backers" target="_blank"><img src="https://opencollective.com/OpenMined/backers.svg?width=890"></a>


## Sponsors

Support this project by becoming a sponsor. Your logo will show up here with a link to your website. [[Become a sponsor](https://opencollective.com/OpenMined#sponsor)]

<a href="https://opencollective.com/OpenMined/sponsor/0/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/0/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/1/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/1/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/2/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/2/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/3/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/3/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/4/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/4/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/5/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/5/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/6/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/6/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/7/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/7/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/8/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/8/avatar.svg"></a>
<a href="https://opencollective.com/OpenMined/sponsor/9/website" target="_blank"><img src="https://opencollective.com/OpenMined/sponsor/9/avatar.svg"></a>



## License
[Apache-2.0](https://github.com/OpenMined/PySyft/blob/master/LICENSE) by OpenMined contributors
