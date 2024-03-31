# Enclave Development

Note: Attestation currently works only in Linux x64

## Building Attestion

```sh
cd pacakges/grid/enclave && \
docker build -f attesation.dockerfile  . -t attestation:0.1
```

## Running Attesation Container

```sh
docker run -it --rm --privileged \
  -v /sys/kernel/security:/sys/kernel/security \
  -v /dev/tpmrm0:/dev/tpmrm0 attest:0.1
```
