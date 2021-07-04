#### INSTRUCTIONS
# BUILD syft first as a base image
# $ cd PySyft
# $ docker build -f docker/syft.Dockerfile --build-arg GPU=false -t openmined/syft:latest -t openmined/syft:`python VERSION` .

# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
FROM python:3.9-slim

# envs and args
ARG GPU=false
ENV LANG C.UTF-8
ENV TORCH_VERSION=1.8.1

# # CUDA
# # RUN apt-get update && apt-get install -y python3 python3-pip

# pydp requires GLIBCXX_3.4.26 from gcc and GLIBC_2.29 from libc6
# strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
# petlib requires python3-dev libssl-dev libffi-dev
# git is required for pip installs from github
RUN echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -yqq \
    gcc gcc-9 g++-9 libc6 git python3-dev libssl-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# update pip
RUN --mount=type=cache,target=/root/.cache python3 -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache python3 -m pip install packaging

# syft
RUN mkdir -p /syft
COPY ./packages/syft /syft

# use CPU torch to keep size down
RUN if [ "$GPU" = "true" ]; then python /syft/scripts/adjust_torch_versions.py /syft/setup.cfg $TORCH_VERSION gpu; fi
# use GPU torch if we want it
RUN if [ "$GPU" = "false" ]; then python /syft/scripts/adjust_torch_versions.py /syft/setup.cfg $TORCH_VERSION; fi

# install all with libs for now
RUN --mount=type=cache,target=/root/.cache cd syft && pip install -e .[all] -f https://download.pytorch.org/whl/torch_stable.html

# install unstable libs
RUN --mount=type=cache,target=/root/.cache pip install -r /syft/requirements.unstable.txt
