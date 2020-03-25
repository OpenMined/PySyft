# PyGrid Node

This app is a Flask app that represents a server that a Grid Client communicates with. We extend this server with the ability to store models and tensors on a redis database and to communicate via HTTP and websockets.

### Start a PyGrid Node locally


#### Using Python
To start a grid node using python, run:
```
python websocket_app.py 
```
You can pass the arguments or use environment variables to set the grid node configs.  

**Arguments**
```
  -h, --help                shows the help message and exit
  --id [ID]                 the grid node identifier, e.g. --id=alice.
  -p [PORT], --port [PORT]  port to run the server on
  --host [HOST]             the grid node host
  --gateway_url [URL]       address used to join a Grid Network.
  --db_url [URL]            REDIS database server address
```

**Environment Variables**
- `ID` - The grid node identifier
- `PORT` -  Port to run server on.
- `ADDRESS` - The grid node address/host
- `REDISCLOUD_URL` - The redis database URL
- `GRID_GATEWAy_URL` - The address used to join a Grid Network

### Docker

The latests PyGrid Node images are available on Docker Hub  

PyGrid Node Docker image - `openmined/grid-node`

#### Pulling images
```
$ docker pull openmined/grid-node  # Download grid node image
```

#### Build your own PyGrid Node image
```
$ docker build openmined/grid-node  # Build grid node image
```
