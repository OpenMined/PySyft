#### INSTRUCTIONS
# BUILD
# $ cd PySyft

# for DEV mode, editable source and hot reloading
# $ docker build -f docker/Dockerfile -t openmined/grid-domain-dev:latest -t openmined/grid-domain-dev:`python VERSION` .
# $ docker run -it -v "`pwd`/packages/grid/apps/domain:/domain" -v "`pwd`/packages/syft:/syft" -p 5000:5000 openmined/grid-domain-dev

# for PROD mode, non editable and smaller
# $ docker build -f docker/Dockerfile --build-arg BUILD_ENV=prod -t openmined/grid-domain:latest -t openmined/grid-domain:`python VERSION` .
# $ docker run -it -p 5000:5000 openmined/grid-domain

# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
FROM python:3.9

# envs
ENV LANG C.UTF-8
ARG BUILD_ENV=dev

# # CUDA
# # RUN apt-get update && apt-get install -y python3 python3-pip

# pydp requires GLIBCXX_3.4.26 from gcc and GLIBC_2.29 from libc6
# strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.28 | grep GLIBCXX
RUN echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -yqq \
    gcc libc6 && \
    rm -rf /var/lib/apt/lists/*

# update pip
RUN --mount=type=cache,target=/root/.cache python3 -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache python3 -m pip install poetry

# syft
RUN mkdir -p /syft
COPY ./packages/syft /syft

ARG GPU=false
ENV TORCH_VERSION=1.8.1

# use CPU torch to keep size down
RUN if [ "$GPU" = "true" ]; then python /syft/scripts/adjust_torch_versions.py /syft/setup.cfg $TORCH_VERSION gpu; fi
# use GPU torch if we want it
RUN if [ "$GPU" = "false" ]; then python /syft/scripts/adjust_torch_versions.py /syft/setup.cfg $TORCH_VERSION; fi

# dev only we want syft source in editable mode
RUN --mount=type=cache,target=/root/.cache if [ "$BUILD_ENV" = "dev" ]; then cd syft && pip install -e .[ci-all] -f https://download.pytorch.org/whl/torch_stable.html; fi

# in prod we build the wheel, install and then cleanup
RUN if [ "$BUILD_ENV" = "prod" ]; then pip install tox && cd /syft && tox -e build; fi
RUN --mount=type=cache,target=/root/.cache if [ "$BUILD_ENV" = "prod" ]; then pip install /syft/dist/*.whl -f https://download.pytorch.org/whl/torch_stable.html; fi
RUN if [ "$BUILD_ENV" = "prod" ]; then rm -rf /syft; fi

# until domain is a wheel we will copy and install dependencies
RUN mkdir -p /domain
COPY ./packages/grid/apps/domain /domain
RUN cd /domain && poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN --mount=type=cache,target=/root/.cache pip install -r /domain/requirements.txt

# remove all layers in the final output to reduce the image size
FROM python:3.9
COPY --from=0 / /

# run the domain in dev mode
CMD ["bash", "-c", "cd /domain && ./dev.sh"]
