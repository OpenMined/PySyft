ARG SEAWEEDFS_VERSION

# FROM chrislusf/seaweedfs:${SEAWEEDFS_VERSION}_large_disk
FROM chrislusf/seaweedfs:3.62_large_disk

WORKDIR /

RUN apk update && \
    apk add --no-cache python3 py3-pip ca-certificates bash

COPY requirements.txt app.py /

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY --chmod=755 start.sh mount_command.sh /

ENTRYPOINT ["/start.sh"]
