# Starting from on the base image, copy the files that are strictly required
# to install PySyft and install it

FROM pysyft-base:local

WORKDIR /PySyft
COPY syft syft
COPY bin bin
COPY Makefile setup.py setup.cfg README.md LICENSE requirements.txt ./
RUN make install
