FROM alpine:edge
RUN ["apk", "add", "--no-cache", "python3", "python3-dev", "musl-dev", "linux-headers", "g++", "gmp-dev", "mpfr-dev", "mpc1-dev", "ca-certificates"]

RUN ["mkdir", "/PySyft"]
COPY requirements.txt /PySyft

WORKDIR /PySyft
RUN ["pip3", "install", "-r", "requirements.txt"]
COPY . /PySyft
RUN ["python3", "setup.py", "install"]
