FROM --platform=$BUILDPLATFORM ubuntu AS build
RUN apt-get -y update && apt-get -y upgrade && apt-get -y dist-upgrade
RUN apt-get install openssl libssl-dev
RUN apt-get -y install python3 g++ python3-dev libffi-dev git gcc g++ musl-dev python3-pip && pip3 install --upgrade pip setuptools
RUN git clone https://github.com/google/jax.git
RUN pip install numpy wheel
RUN python3 jax/build/build.py --target_cpu_features default
RUN pip install dist/*.whl

