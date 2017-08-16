# Syft

> Homomorphically Encrypted Deep Learning Library

The goal of this library is to give the user the ability to efficiently train Deep Learning models in a homomorphically encrypted state without needing to be an expert in either. Furthermore, by understanding the characteristics of both Deep Learning and Homomorphic Encryption, we hope to find very performant combinations of the two.  See [notebooks](./notebooks) folder for tutorials on how to use the library.

## Installation

You need to install this library locally before running any of the notebooks this repository or the [main demonstration](https://github.com/OpenMined/sonar):

```sh
# Get dependencies ready
pip install -r requirements.txt
# install the lib locally
python setup.py install
```

### For Anaconda Users:

```
bash install_for_anaconda_users.sh
```
**Windows**
```sh
conda install -c conda-forge gmpy2
pip install -r requirements.txt
python setup.py install
```

### For Docker Users

Install Docker from https://www.docker.com/
For macOS users with [Homebrew](https://brew.sh/) installed, use `brew cask install docker`

Then, run:

```sh
git clone https://github.com/OpenMined/PySyft.git
cd PySyft/notebooks/
docker run --rm -it -v $PWD:/notebooks -w /notebooks -p 8888:8888 openmined/pysyft jupyter notebook --ip=0.0.0.0 --allow-root
```

## For Contributors

If you are interested in contributing to Syft, first check out our [Contributor Quickstart Guide](https://github.com/OpenMined/Docs/blob/master/contributing/quickstart.md) and then checkout our [Project Roadmap](https://github.com/OpenMined/Syft/blob/master/ROADMAP.md) and sign into our Slack Team channel #syft to let us know which projects sound interesting to you! (or propose your own!).

## Running tests

```sh
cd PySyft
pytest
```

## Relevant Literature

As both Homomorphic Encryption and Deep Learning are still somewhat sparsely known, below is a curated list of relevant reading materials to bring you up to speed with the major concepts and themes of these exciting fields.

### Encrypted Deep Learning - Recommended Reading:
- How to build a fully encrypted AI model (trained on unencrypted data):  
  - http://iamtrask.github.io/2017/03/17/safe-ai/
- Simple secure protocol for federated machine learning (using a python-paillier library):  
  - https://blog.n1analytics.com/distributed-machine-learning-and-partially-homomorphic-encryption-1/
- Prototype for using encrypted AI to preserve user privacy (in python):  
  - http://iamtrask.github.io/2017/06/05/homomorphic-surveillance/
- Manual for Using Homomorphic Encryption for Bioinformatics (paper):  
  - https://www.microsoft.com/en-us/research/wp-content/uploads/2015/11/ManualHE-3.pdf

### Homomorphic Encryption - Recommended Reading:
- https://eprint.iacr.org/2014/062.pdf
- http://heat-h2020-project.blogspot.co.uk/2017/02/homomorphic-encryption-api-software.html
- http://heat-h2020-project.blogspot.co.uk/2017/

### Relevant Papers:
- http://proceedings.mlr.press/v54/esperanca17a/esperanca17a.pdf
- https://sites.google.com/view/psml/program/abstracts
- https://arxiv.org/pdf/1508.06845.pdf
- https://arxiv.org/pdf/1508.06574.pdf
- https://eprint.iacr.org/2016/892
- https://www1.cs.fau.de/filepool/publications/octavian_securescm/secfp-fc10.pdf

### Related Libraries:
- http://www.louisaslett.com/HomomorphicEncryption/#details
- https://github.com/iamtrask/linreg-mpc
- https://github.com/dask/dask-tutorial

### Related Blogs:
- https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
- https://mortendahl.github.io/2017/06/04/secret-sharing-part1/
- https://mortendahl.github.io/2017/06/24/secret-sharing-part2/
- https://blog.n1analytics.com/distributed-machine-learning-and-partially-homomorphic-encryption-1/
- https://blog.mturk.com/tutorial-how-to-verify-crowdsourced-training-data-using-a-known-answer-review-policy-85596fb55ed
