![PyGrid logo](https://raw.githubusercontent.com/OpenMined/design-assets/master/logos/PyGrid/horizontal-primary-trans.png)

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/OpenMined/PyGrid/dev) [![Run Tests](https://github.com/OpenMined/PyGrid/workflows/Run%20tests/badge.svg?branch=dev)](https://github.com/OpenMined/PyGrid/actions?query=workflow%3A%22Run+tests%22) [![Docker build](https://github.com/OpenMined/PyGrid/workflows/Docker%20build/badge.svg)](https://github.com/OpenMined/PyGrid/actions?query=workflow%3A%22Docker+build%22) [![Chat on Slack](https://img.shields.io/badge/chat-on%20slack-7A5979.svg)](https://openmined.slack.com/messages/team_pysyft) [![FOSSA Status](https://camo.githubusercontent.com/c0cb82174c3eb8fcbb00a46eb237556f63b36804/68747470733a2f2f6170702e666f7373612e696f2f6170692f70726f6a656374732f6769742532426769746875622e636f6d2532466d6174746865772d6d6361746565722532465079537966742e7376673f747970653d736d616c6c)](https://app.fossa.io/projects/git%2Bgithub.com%2Fmatthew-mcateer%2FPySyft?ref=badge_small)

PyGrid is a peer-to-peer network of data owners and data scientists who can collectively train AI models using [PySyft](https://github.com/OpenMined/PySyft/).


## Overview
- [Overview](#overview)
- [Architecture](#architecture)
- [Getting started](#getting-started)
    - [Build Grid Platform Locally](#start-grid-platform-locally)
    - [Build images](#build-images)
    - [Let's put all together](#lets-put-all-together)
- [Try out the Tutorials](#try-out-the-tutorials)
- [Start Contributing](#start-contributing)
- [High-level Architecture](#high-level-architecture)
- [Disclaimer](#disclaimer)
- [License](#license)

## Architecture
PyGrid platform is composed by three different components.

**PyGrid App** - A Flask based application used to manage/monitor/control and route grid Nodes/Workers remotely.  
**Grid Nodes** - Server based apps used to store and manage data access in a secure and private way.  
**Grid Workers** - Client based apps that uses different Syft based libraries to perform federated learning (ex: syft.js, KotlinSyft, SwiftSyft).

## Getting started
To boot the entire PyGrid platform locally, we will use docker containers.
To install docker the dependencies, just follow [docker documentation](https://docs.docker.com/install/).

### Start Grid platform locally

#### 1 - Using Docker

The latest PyGrid Gateway and Node images are available on the Docker Hub.
- PyGrid - `openmined/grid-gateway`
- Grid Node - `openmined/grid-node`

##### 1.1 - Setting the Domain Names

Before start the grid platform locally using docker, we need to set up the domain names used by the bridge network. In order to use these nodes from outside of containers context, you should add the following domain names on your `/etc/hosts`
```
127.0.0.1 gateway
127.0.0.1 bob
127.0.0.1 alice
127.0.0.1 bill
127.0.0.1 james
```

#### 1.2 - Run Docker Images
To setup and start the PyGrid platform you just need start the docker-compose process.
```
$ docker-compose up
```

It will download the latest openmined's docker images and start a grid platform with 1 gateway and 4 grid nodes.  
**PS:** Feel free to increase/decrease the number of initial PyGrid nodes ***(you can do this by changing the docker-compose.yml file)***.

### 1.3 - Build your own images (Optional)
```
$ docker build -t openmined/grid-node ./app/websocket/  # Build PyGrid node image
$ docker build -t openmined/grid-gateway ./gateway/  # Build gateway image
```


#### 2 - Starting manually
To start the PyGrid app manually, run:

```
python grid.py 
```
You can pass the arguments or use environment variables to set the gateway configs.  

**Arguments**
```
  -h, --help                shows the help message and exit
  -p [PORT], --port [PORT]  port to run server on (default: 5000)
  --host [HOST]             the grid gateway host
  --num_replicas            the number of replicas to provide fault tolerance to model hosting
  --start_local_db          if this flag is used a SQLAlchemy DB URI is generated to use a local db
```

**Environment Variables**
- `GRID_GATEWAY_PORT` -  Port to run server on.
- `GRID_GATEWAY_HOST` - The grid gateway host
- `NUM_REPLICAS` - Number of replicas to provide fault tolerance to model hosting
- `DATABASE_URL` - The gateway database URL
- `SECRET_KEY` - The secret key

#### For development purposes
You can also start the PyGrid app by running the `dev_server.sh` script.
```
$ ./dev_server.sh
```
This script uses the `dev_server.conf.py` as configuration file, including some gunicorn preferences and environment variables. The file is pre-populated with the default environment variables. You can set them by editing the following property:
```python
raw_env = [
    'PORT=5000',
    'SECRET_KEY=ineedtoputasecrethere',
    'DATABASE_URL=sqlite:///databasegateway.db',
]
```

### Kubernetes deployment.
You can now deploy the PyGrid app and Grid Node docker containers on kubernetes. This can be either to a local (minikube) cluster or a remote cluster (GKE, EKS, AKS etc). The steps to setup the cluster can be found in [./k8s/Readme.md](https://github.com/OpenMined/PyGrid/tree/dev/k8s)

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
