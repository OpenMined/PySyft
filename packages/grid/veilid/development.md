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

#### 2. Production Mode

```sh
cd packages/grid/veilid && \
docker run --rm  -p 4000:4000 -p 5959:5959 -p 5959:5959/udp veilid:0.1
```
