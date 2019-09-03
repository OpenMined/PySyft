FROM python:3.6-slim

ENV WORKER_DIR /worker

# Install PySyft
RUN apt-get update && apt-get install -y gcc
RUN pip install --no-cache-dir syft

# Add worker code
WORKDIR $WORKER_DIR
COPY ./entrypoint.sh .
RUN chmod +x ./entrypoint.sh
COPY ./worker-server.py .
EXPOSE 8777

ENTRYPOINT "$WORKER_DIR/entrypoint.sh"
