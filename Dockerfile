FROM alpine:edge
RUN apk add --no-cache \
  python3 python3-dev \
  g++ linux-headers \
  musl-dev gmp-dev mpfr-dev mpc1-dev \
  ca-certificates openblas-dev \
  make gfortran

WORKDIR /PySyft
COPY . .

RUN pip3 install -r requirements.txt
RUN python3 setup.py install
