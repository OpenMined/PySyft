# Enclave Development

Note: Attestation currently works only in Linux x64

## Building Attestion

```sh
cd packages/grid/enclave/attestation && \
docker build -f attesation.dockerfile  . -t attestation:0.1
```

## For CPU Attestation

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

## For GPU Attestation

We would need to install Nvidia Container Toolkit on host system and ensure we have CUDA Drivers installed.
Link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

We could need to modify the docker run command as

```sh
docker run -it --rm --privileged --gpus all --runtime=nvidia \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attestation:0.1
```

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
