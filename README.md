![PyGrid logo](https://raw.githubusercontent.com/OpenMined/design-assets/master/logos/PyGrid/horizontal-primary-trans.png)

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/OpenMined/PyGrid/dev) [![Run Tests](https://github.com/OpenMined/PyGrid/workflows/Run%20tests/badge.svg?branch=dev)](https://github.com/OpenMined/PyGrid/actions?query=workflow%3A%22Run+tests%22) [![Docker build](https://github.com/OpenMined/PyGrid/workflows/Docker%20build/badge.svg)](https://github.com/OpenMined/PyGrid/actions?query=workflow%3A%22Docker+build%22) [![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://openmined.slack.com/messages/team_pysyft) [![FOSSA Status](https://camo.githubusercontent.com/c0cb82174c3eb8fcbb00a46eb237556f63b36804/68747470733a2f2f6170702e666f7373612e696f2f6170692f70726f6a656374732f6769742532426769746875622e636f6d2532466d6174746865772d6d6361746565722532465079537966742e7376673f747970653d736d616c6c)](https://app.fossa.io/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft?ref=badge_small)

PyGrid is a peer-to-peer network of data owners and data scientists who can collectively train AI models using [PySyft](https://github.com/OpenMined/PySyft/).


## Overview
- [Overview](#overview)
- [Getting started](#getting-started)
    - [Build Grid Platform Locally](#start-grid-platform-locally)
    - [Build images](#build-images)
    - [Let's put all together](#lets-put-all-together)
- [Try out the Tutorials](#try-out-the-tutorials)
- [Start Contributing](#start-contributing)
- [High-level Architecture](#high-level-architecture)
- [Disclaimer](#disclaimer)
- [License](#license)


## Getting started
To boot the entire PyGrid platform locally, we will use docker containers.
To install docker the dependencies, just follow [docker documentation](https://docs.docker.com/install/).

### Start Grid platform locally

#### Using Docker

The latest PyGrid Gateway and Node images are available on the Docker Hub.
- PyGrid Gateway - `openmined/grid-node`
- PyGrid Node - `openmined/grid-node`

###### Setting the Domain Names

Before start the grid platform locally using docker, we need to setup the domain names used by the bridge network. In order to access these nodes from outside of containers context, you need to work-around by adding the following domain names on your `/etc/hosts`
```
127.0.0.1 gateway
127.0.0.1 bob
127.0.0.1 alice
127.0.0.1 bill
127.0.0.1 james
```


It will download the latest openmined's docker images and start a grid platform with 1 gateway and 4 grid nodes.
**PS:** Feel free to increase/decrease the number of initial PyGrid nodes ***(you can do this by changing the docker-compose.yml file)***.
```
$ docker-compose up
```
If you want to rebuild and run the images, you just need to add the `--build` param when running `docker-compose up`
```
$ docker-compose up --build
```


#### Starting manually
Start the grid platform manually with 1 gateway and how many grid nodes you want.  

- **PyGrid Gateway** - Check out the instructions under [`/gateway`](./gateway)

- **PyGrid Node** - Check out the instructions under [`/app/websocket`](./app/websocket)

### Kubernetes deployment.
You can now deploy the grid-gateway and grid-node docker containers on kubernetes. This can be either to a local (minikube) cluster or a remote cluster (GKE, EKS, AKS etc). The steps to setup the cluster can be found in [./k8s/Readme.md](https://github.com/OpenMined/PyGrid/tree/dev/k8s)

### Build your own images
```
$ docker build -t openmined/grid-node ./app/websocket/  # Build PyGrid node image
$ docker build -t openmined/grid-gateway ./gateway/  # Build gateway image
```

## Try out the Tutorials
A comprehensive list of tutorials can be found [here](https://github.com/OpenMined/PySyft/tree/master/examples/tutorials/grid).

These tutorials cover how to create a PyGrid node and what operations you can perform.

## Start Contributing
The guide for contributors can be found [here](https://github.com/OpenMined/PyGrid/tree/dev/CONTRIBUTING.md). It covers all that you need to know to start contributing code to PyGrid in an easy way.

Also join the rapidly growing community of 7300+ on [Slack](http://slack.openmined.org). The slack community is very friendly and great about quickly answering questions about the use and development of PyGrid/PySyft!

We also have a Github Project page for a Federated Learning MVP [here](https://github.com/orgs/OpenMined/projects/13).  
You can check the PyGrid's official development and community roadmap [here](https://github.com/OpenMined/Roadmap/tree/master/pygrid_team).

## High-level Architecture

![High-level Architecture](https://raw.githubusercontent.com/OpenMined/PyGrid/dev/art/PyGrid-Arch.png)


## Disclaimer
Do ***NOT*** use this code to protect data (private or otherwise) - at present it is very insecure.

## License

[Apache License 2.0](https://github.com/OpenMined/PyGrid/blob/dev/LICENSE)
