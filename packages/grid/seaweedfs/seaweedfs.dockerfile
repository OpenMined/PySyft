ARG SEAWEEDFS_VERSION="3.64"
FROM chrislusf/seaweedfs:${SEAWEEDFS_VERSION}_large_disk

WORKDIR /root/swfs

RUN apk update && \
    apk add --no-cache python3 py3-pip curl openssl envsubst

COPY ./requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

RUN mkdir -p /data/master/ /data/vol/blob/ /data/vol/idx/ && \
    ulimit -n 10240

COPY ./scripts ./scripts
COPY ./src ./src
COPY ./config/app/ .
COPY ./config/seaweedfs/ /etc/seaweedfs/
COPY ./config/s3config.template.json /etc/secrets/

ENV \
    # master
    SWFS_MASTER_DIR="/data/master/" \
    SWFS_VOLUME_SIZE_LIMIT_MB="1000" \
    # filer
    SWFS_FILER_CHUNKS_MB="64" \
    SWFS_FILER_UPLOAD_LIMIT_MB="0" \
    # volume
    SWFS_VOLUME_DIR="/data/vol/blob/" \
    SWFS_VOLUME_IDX_DIR="/data/vol/idx/" \
    SWFS_VOLUME_MAX="0" \
    SWFS_VOLUME_INDEX="leveldb" \
    SWFS_VOLUME_UPLOAD_LIMIT_MB="0" \
    SWFS_VOLUME_DOWNLOAD_LIMIT_MB="0" \
    SWFS_UNCACHE_MINAGE="86400" \
    # mount api
    MOUNT_API_PORT="4001" \
    UVICORN_LOG_LEVEL="info"

# overrides base image entrypoint
ENTRYPOINT ["sh"]
CMD ["./scripts/start.sh"]
