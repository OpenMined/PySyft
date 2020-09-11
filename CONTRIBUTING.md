# Contributor Guidelines

## Getting Started

If you're going to be contributing to PySyft, you'll want to have a fast developer cycle and the ability to experience how your code will be used in a data science context. For this, we recommend the following development flow.

Step 1) Uninstall your default version of PySyft

pip uninstall syft

Step 2) Install Syft as a folder backed reference. From the root of the PySyft repository run the following.

pip install -e .

Step 3) Launch jupyter notebook

jupyter lab

Step 4) experiment within notebooks with what you wnat the end data scientist experience to be and then merge code in to the codebase as you go.

### Git and Github

All our development is done using Git and Github. If you're not too familiar with Git and Github, [start by reviewing this guide](https://guides.github.com/activities/hello-world).

### Slack

[Join our Slack community](http://slack.openmined.org).

### Setting up Pre-Commit Hook

PySyft uses the python package `pre-commit` to make sure the correct formatting (black & flake) is applied.

You can install it via `pip install pre-commit`

Then you just need to call `pre-commit install`
