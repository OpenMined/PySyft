## Veilid - Development Instructions

### 1. Building Veilid Container

```sh
cd packages/grid/veilid && docker build -f veilid.dockerfile -t veilid:0.1 .
```

### Running veilid Container

#### 1. Development Mode

```sh
cd packages/grid/veilid && \
docker run --rm -e DEV_MODE=True -p 4000:4000 -p 5959:5959 -p 5959:5959/udp  -v $(pwd)/server:/app/server veilid:0.1
```

##### 2. Additional Flags for Development

```
a. VEILID_FLAGS="--debug" (For Veilid Debug logs)
b. APP_LOG_LEVEL="debug" (For changing logging method inside the application could be info, debug, warning, critical)
c. UVICORN_LOG_LEVEL="debug" (For setting logging method for uvicorn)
```

#### 3. Production Mode

```sh
cd packages/grid/veilid && \
docker run --rm  -p 4000:4000 -p 5959:5959 -p 5959:5959/udp veilid:0.1
```

### Kubernetes Development

#### 1. Gateway Node

##### Creation

```sh
bash -c '\
    export CLUSTER_NAME=testgateway1 CLUSTER_HTTP_PORT=9081 DEVSPACE_PROFILE=gateway && \
    tox -e dev.k8s.start && \
    tox -e dev.k8s.hotreload'
```

##### Deletion

```sh
bash -c "CLUSTER_NAME=testgateway1 tox -e dev.k8s.destroy || true"
```

#### 2. Domain Node

##### Creation

```sh
bash -c '\
    export CLUSTER_NAME=testdomain1 CLUSTER_HTTP_PORT=9082 && \
    tox -e dev.k8s.start && \
    tox -e dev.k8s.hotreload'
```

##### Deletion

```sh
bash -c "CLUSTER_NAME=testdomain1 tox -e dev.k8s.destroy || true"
```
