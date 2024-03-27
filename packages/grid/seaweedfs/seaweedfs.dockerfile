ARG SEAWEEDFS_VERSION="3.64"

FROM chrislusf/seaweedfs:${SEAWEEDFS_VERSION}_large_disk

WORKDIR /root/swfs

RUN apk update && \
    apk add --no-cache python3 py3-pip ca-certificates bash supervisor nano

ENV \
    S3_PORT="8333" \
    S3_VOLUME_SIZE_MB="1024" \
    S3_CONFIG_PATH="/root/swfs/s3_config.json" \
    S3_MOUNT_DIRS="/data/vol0/" \
    S3_MOUNT_DIRS_MAX="0" \
    SEAWEED_MOUNT_PORT="4001" \
    UVICORN_LOG_LEVEL="info"

COPY ./requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    mkdir -p /data/master/ /data/vol0/

COPY ./scripts ./scripts
COPY ./src ./src
COPY supervisord.conf filer.toml .

# overrides base image entrypoint
ENTRYPOINT ["bash"]
CMD ["./scripts/start.sh"]
