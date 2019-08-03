FROM python:3.6-slim

ENV WORKER_DIR /worker

# Install PySyft
RUN apt-get update && apt-get install -y gcc
RUN pip install --no-cache-dir syft

# TODO: Add worker code
RUN mkdir $WORKER_DIR
WORKDIR $WORKER_DIR
