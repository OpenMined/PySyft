FROM tailscale/tailscale:v1.29.125

RUN apk update && apk upgrade --available

RUN --mount=type=cache,target=/var/cache/apk \
    apk -U upgrade || true; \
    apk fix || true; \
    apk add --no-cache python3 py3-pip ca-certificates bash || true

WORKDIR /tailscale
COPY ./requirements.txt /tailscale/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt

COPY ./tailscale.sh /tailscale/tailscale.sh
COPY ./tailscale.py /tailscale/tailscale.py

ENV HOSTNAME="node"

CMD ["sh", "-c", "/tailscale/tailscale.sh ${HOSTNAME}"]
