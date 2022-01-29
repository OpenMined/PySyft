FROM tailscale/tailscale:v1.16

RUN --mount=type=cache,target=/var/cache/apk \
    apk add --no-cache python3 py3-pip ca-certificates

WORKDIR /tailscale
COPY ./requirements.txt /tailscale/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt

COPY ./tailscale.sh /tailscale/tailscale.sh
COPY ./tailscale.py /tailscale/tailscale.py

ENV HOSTNAME="node"

CMD ["sh", "-c", "/tailscale/tailscale.sh ${HOSTNAME}"]
