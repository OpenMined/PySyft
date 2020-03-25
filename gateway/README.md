# PyGrid Gateway

PyGrid Gateway is a Flask based application used to manage/monitor/control and route grid workers remotely.

## Getting Started

### Start PyGrid Gateway locally


#### Using Python
To start a grid gateway using python, run:
```
python gateway.py 
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
You can also start a gateway using gunicorn by running the `dev_server.sh` script.
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

### Docker

The latests PyGrid Gateway images are available on Docker Hub  

PyGrid Gateway Docker image - `openmined/grid-gateway`

#### Pulling images
```
$ docker pull openmined/grid-gateway  # Download gateway image
```

#### Build your own PyGrid Gateway image
```
$ docker build openmined/grid-gateway  # Build gateway image
```
