ARG SEAWEEDFS_VERSION

FROM chrislusf/seaweedfs:${SEAWEEDFS_VERSION}

WORKDIR /

RUN apk update && \
    apk add --no-cache python3 py3-pip ca-certificates bash

COPY requirements.txt app.py /
RUN pip install --no-cache-dir -r requirements.txt

COPY --chmod=755 start.sh mount_command.sh /

ENTRYPOINT ["/start.sh"]
