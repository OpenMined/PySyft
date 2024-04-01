ARG SEAWEEDFS_VERSION="3.64"

FROM chrislusf/seaweedfs:${SEAWEEDFS_VERSION}_large_disk

WORKDIR /root/swfs

RUN apk update && \
    apk add --no-cache python3 py3-pip ca-certificates supervisor curl

COPY ./requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

RUN mkdir -p /data/master/ /data/vol/ /data/vol_idx/ /data/mount/ /data/mount/creds/ && \
    ulimit -n 10240

COPY ./scripts ./scripts
COPY ./config/ .
COPY ./src ./src

ENV \
    SWFS_MASTER_DIR="/data/master/" \
    SWFS_VOLUME_SIZE_LIMIT_MB="1000" \
    SWFS_VOLUME_DIR="/data/vol/" \
    SWFS_VOLUME_IDX_DIR="/data/vol_idx/" \
    SWFS_VOLUME_MAX="0" \
    S3_CONFIG_PATH="/root/swfs/config/s3_config.json" \
    S3_ROOT_USER="admin" \
    S3_ROOT_PASSWORD="admin" \
    MOUNT_API_PORT="4001" \
    UVICORN_LOG_LEVEL="info"

# overrides base image entrypoint
ENTRYPOINT ["sh"]
CMD ["./scripts/start.sh"]
