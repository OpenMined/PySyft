# Enclave Development

## Building Attestion Containers

NOTE: Even on Arm machines, we build x64 images.
As some dependent packages in the dockerfile do not have arm64 equivalent.
It would take 10 minutes to build the image in emulation for the first time
in Arm machines.After which , the subsequent builds would be instant.

```sh
cd packages/grid/enclave/attestation && \
docker build -f attestation.dockerfile  . -t attestation:0.1 --platform linux/amd64
```

## Running the container in development mode

```sh
cd packages/grid/enclave/attestation && \
docker run -it --rm -e DEV_MODE=True -p 4455:4455 -v $(pwd)/server:/app/server attestation:0.1
```

## For fetching attestation report by FastAPI

### CPU Attestation

```sh
docker run -it --rm --privileged \
  -p 4455:4455 \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1
```

```sh
curl localhost:4455/attest/cpu
```

### GPU Attestation

#### Nvidia GPU Requirements

We would need to install Nvidia Container Toolkit on host system and ensure we have CUDA Drivers installed.
Link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

```sh
docker run -it --rm --privileged --gpus all --runtime=nvidia \
  -p 4455:4455 \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1
```

```sh
curl localhost:4455/attest/gpu
```

## For fetching attestation report directly by docker

### CPU Attestation

```sh
docker run -it --rm --privileged \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1 /bin/bash
```

In the shell run

```sh
./AttestationClient
```

This would return either True or False indicating status of attestation

This could also be customized with Appraisal Policy

To retrieve JWT from Microsoft Azure Attestation (MAA)

```sh
./AttestationClient -o token
```

### For GPU Attestation

```sh
docker run -it --rm --privileged --gpus all --runtime=nvidia \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1 /bin/bash
```

Invoke python shell
In the python shell run

```python3
from nv_attestation_sdk import attestation


NRAS_URL="https://nras.attestation.nvidia.com/v1/attest/gpu"
client = attestation.Attestation()
client.set_name("thisNode1")
client.set_nonce("931d8dd0add203ac3d8b4fbde75e115278eefcdceac5b87671a748f32364dfcb")
print ("[RemoteGPUTest] node name :", client.get_name())

client.add_verifier(attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, "")
client.attest()
```
