FROM shaynesweeney/tailscale:latest

RUN --mount=type=cache,target=/var/cache/apk \
    apk add --no-cache python3 py3-pip ca-certificates

WORKDIR /tailscale
COPY ./requirements.txt /tailscale/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt

COPY ./tailscale.sh /tailscale/tailscale.sh
COPY ./tailscale.py /tailscale/tailscale.py
COPY ./secure/__init__.py /tailscale/secure/__init__.py
COPY ./secure/api.py /tailscale/secure/api.py
COPY ./secure/base_entrypoint.py /tailscale/secure/base_entrypoint.py

ENV HOSTNAME="node"

CMD ["sh", "-c", "/tailscale/tailscale.sh ${HOSTNAME}"]
