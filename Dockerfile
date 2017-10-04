FROM alpine:edge
RUN apk add --no-cache python3 python3-dev \
                    musl-dev linux-headers g++ \
                    gmp-dev mpfr-dev mpc1-dev \
                    ca-certificates openblas-dev gfortran

RUN mkdir /src

ENV PYSYFT_DIR /src/PySyft

## PySyft Dependency
RUN mkdir $PYSYFT_DIR
WORKDIR $PYSYFT_DIR
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
RUN python3 setup.py install

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
