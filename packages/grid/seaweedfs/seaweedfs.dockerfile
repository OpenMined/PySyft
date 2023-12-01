FROM chrislusf/seaweedfs:3.57

WORKDIR /

RUN apk update && \
    apk upgrade --available && \
    apk add --no-cache python3 py3-pip ca-certificates bash

COPY requirements.txt app.py /
RUN pip install --no-cache-dir -r requirements.txt

COPY --chmod=755 start.sh mount_command /

ENTRYPOINT ["/usr/bin/env"]
CMD ["bash", "./start.sh"]
