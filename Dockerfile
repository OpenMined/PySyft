FROM alpine:edge
RUN ["apk", "add", "--no-cache", "python3", "python3-dev", "musl-dev", "linux-headers", "g++", "gmp-dev", "mpfr-dev", "mpc1-dev"]

RUN ["mkdir", "/syft"]
COPY requirements.txt /syft

WORKDIR /syft
RUN ["pip3", "install", "-r", "requirements.txt"]
COPY . /syft
RUN ["python3", "setup.py", "install"]