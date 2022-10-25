FROM python:3.10.7-slim as build
RUN apt-get -y update && apt-get -y upgrade && apt-get -y dist-upgrade
RUN apt-get -y install \
    g++ \
    gcc \
    git \
    libffi-dev \
    libssl-dev \
    musl-dev \
    openssl \
    python3 \
    python3-dev \
    python3-pip
RUN pip3 install --upgrade pip setuptools numpy wheel
RUN git clone https://github.com/google/jax.git

CMD ["bash"]

# RUN python3 jax/build/build.py --target_cpu_features default
# RUN pip3 install dist/*.whl
