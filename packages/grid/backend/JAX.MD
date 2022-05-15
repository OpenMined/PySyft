# Building jaxlib for linux Arm64 to run in Docker on Apple Silicon

Create a high CPU linux x86 machine. I used 32 CPUs on Azure and compilation took 4.5 hours.

Once the latest docker is setup run the following steps.

## Create a buildx builder for arm64

```
$ docker buildx create --platform linux/arm64 --name arm64builder
$ docker buildx use arm64builder
```

## Install the QEMU arm64 emulator layers so you can run the container

```
$ docker run --privileged --rm tonistiigi/binfmt --install arm64
$ docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

## Build the dockerfile using arm64

```
$ docker buildx build --platform linux/arm64 -f build_jax.dockerfile . --output "type=docker,name=build_jax_arm64:latest"
```

## Run the container to start the build

```
$ mkdir bazel_cache
$ docker run -it --platform linux/arm64 -v `pwd`/bazel_cache:/root/.cache build_jax_arm64 bash
```

## Verify you are running arm64

```
$ uname -a
Linux a875c1c357a3 5.13.0-1017-azure #19~20.04.1-Ubuntu SMP Mon Mar 7 11:34:26 UTC 2022 aarch64 GNU/Linux
```

## Confirm your GLIBC version

If this doesn't match the environment you will be running the jaxlib wheel it might fail.
In this case we are using the same `FROM python:3.10.4-slim as build` to build as jaxlib we do in our production container to run it.

```
$ ldd --version
ldd (Debian GLIBC 2.31-13+deb11u3) 2.31
```

## Checkout the version you want to build

```
$ cd jax
$ git fetch --all --tags
$ git checkout tags/jaxlib-v0.3.7 -b jaxlib-v0.3.7
```

## Start the build

```
$ python3 /jax/build/build.py --target_cpu_features default
```

## Check it worked

```
$ pip install dist/*.whl
```

## Copy out the wheel

```
$ pip install --user magic-wormhole
$ /root/.local/bin/wormhole send /jax/build/dist/jaxlib-0.3.7-cp310-none-manylinux2014_aarch64.whl
# run wormhole recieve on the other end
```
