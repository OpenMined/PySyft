## GPU Support - K3d

This document details, on how to enable gpu support in PySyft.

### 1. Step 0: Building k3s Image

Perform this step only when creating new base k3s image or skip this step.

This was tested with k3d guide version: 5.7.2

First , follow this link to create a GPU-based k3s image
https://k3d.io/v5.7.2/usage/advanced/cuda/

Build the image locally.
When building on MacOS , modify the build.sh script to have
`docker --platform linux/amd64,linux/arm64 ....`
and also enable containerd image store in docker desktop settings.

Finally push the image to docker hub

### Step 2: Launch Enclave with GPU

```sh
GPU_ENABLED=true tox -e dev.k8s.launch.enclave
```
