FROM python:3.10.7-slim as build
RUN apt-get -y update --allow-insecure-repositories
RUN apt-get -y upgrade
RUN apt-get -y dist-upgrade
RUN apt-get -y install git
RUN git clone https://github.com/tensorflow/compression.git /tensorflow_compression
WORKDIR /tensorflow_compression
RUN git checkout tags/v2.10.0 -b v2.10.0
RUN apt-get -y install wget
RUN wget https://raw.githubusercontent.com/OpenMined/PySyft/dev/packages/grid/backend/wheels/dm-tree-0.1.7.tar.gz
RUN tar -xf dm-tree-0.1.7.tar.gz --strip-components=6
RUN mv site-packages/* /usr/local/lib/python3.10/site-packages
RUN python -m pip install -U pip setuptools wheel
RUN python -m pip install scipy --only-binary=:all:
RUN python -m pip install tensorflow-probability~=0.15
RUN python -m pip install tensorflow~=2.10.0
RUN wget https://github.com/bazelbuild/bazel/releases/download/5.3.1/bazel-5.3.1-linux-arm64
RUN mv bazel-5.3.1-linux-arm64 bazel
RUN chmod +x bazel
RUN export PATH=`pwd`:$PATH
RUN apt-get -y install gcc g++
RUN ./bazel build -c opt :build_pip_pkg --verbose_failures
RUN python build_pip_pkg.py bazel-bin/build_pip_pkg.runfiles/tensorflow_compression /tmp/tensorflow_compression 2.10.0
