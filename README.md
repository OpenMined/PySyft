![PyGrid logo](https://raw.githubusercontent.com/OpenMined/design-assets/master/logos/PyGrid/horizontal-primary-trans.png)

[![Run Tests](https://github.com/OpenMined/PyGrid/workflows/Run%20tests/badge.svg)](https://github.com/OpenMined/PyGrid/actions?query=workflow%3A%22Run+tests%22) [![Docker build](https://github.com/OpenMined/PyGrid/workflows/Docker%20build/badge.svg)](https://github.com/OpenMined/PyGrid/actions?query=workflow%3A%22Docker+build%22)

PyGrid is a peer-to-peer network of data owners and data scientists who can collectively train AI models using [PySyft](https://github.com/OpenMined/PySyft/). PyGrid is also the central server for conducting both model-centric and data-centric federated learning.

## Architecture

PyGrid platform is composed by three different components.

- **Network** - A Flask-based application used to manage, monitor, control, and route instructions to various PyGrid Nodes.
- **Node** - A Flask-based application used to store private data and models for federated learning, as well as to issue instructions to various PyGrid Workers.
- **Worker** - An emphemeral instance to compute data (managed by PyGrid Node).

![PyGrid Architecture](https://github.com/OpenMined/Roadmap/blob/master/federated_learning/projects/images/new-workflow-network.png?raw=true)

## Getting started

To boot the entire PyGrid platform locally, we will use docker containers. To install Docker, just follow the [docker documentation](https://docs.docker.com/install/).

### Docker

#### 1. Using Docker

The latest PyGrid Network and Node images are available on the Docker Hub.

- PyGrid Network - `openmined/grid-network`
- PyGrid Node - `openmined/grid-node`

#### 2. Setting the Domain Names

Before start the grid platform locally using docker, we need to set up the domain names used by the bridge network. In order to use these nodes from outside of containers context, you should add the following domain names on your `/etc/hosts`

```
127.0.0.1 network
127.0.0.1 bob
127.0.0.1 alice
127.0.0.1 bill
127.0.0.1 james
```

#### 3. Run Docker Images

To setup and start the PyGrid platform you just need start the docker-compose process.

```
$ docker-compose up
```

It will download the latest openmined's docker images and start a grid platform with 1 network and 4 grid nodes.  
**PS:** Feel free to increase/decrease the number of initial PyGrid nodes **_(you can do this by changing the docker-compose.yml file)_**.

### 4. Build your own images (Optional)

```
$ docker build -t openmined/grid-node ./app/websocket/  # Build PyGrid node image
$ docker build -t openmined/grid-network ./network/  # Build network image
```

### Manual Start

To start the PyGrid Node manually, run:

```
cd ./apps/node
./run.sh --port 5000 --start_local_db
```

You can pass the arguments or use environment variables to set the network configs.

**Arguments**

```
  -h, --help                shows the help message and exit
  -p [PORT], --port [PORT]  port to run server on (default: 5000)
  --host [HOST]             the grid network host
  --num_replicas            the number of replicas to provide fault tolerance to model hosting
  --start_local_db          if this flag is used a SQLAlchemy DB URI is generated to use a local db
```

**Environment Variables**

- `GRID_NETWORK_PORT` - Port to run server on.
- `GRID_NETWORK_HOST` - The grid network host
- `NUM_REPLICAS` - Number of replicas to provide fault tolerance to model hosting
- `DATABASE_URL` - The network database URL
- `SECRET_KEY` - The secret key

## Support

For support in using this library, please join the **#lib_pygrid** Slack channel. If you’d like to follow along with any code changes to the library, please join the **#code_pygrid** Slack channel. [Click here to join our Slack community!](https://slack.openmined.org)

## License

[Apache License 2.0](https://github.com/OpenMined/PyGrid/blob/dev/LICENSE)
