FROM shaynesweeney/tailscale:latest

RUN --mount=type=cache,target=/var/cache/apk \
    apk add --no-cache python3 py3-pip

WORKDIR /tailscale
COPY grid/vpn/requirements.txt /tailscale/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt

COPY grid/vpn/tailscale.sh /tailscale/tailscale.sh
COPY grid/vpn/tailscale.py /tailscale/tailscale.py
COPY grid/vpn/secure/__init__.py /tailscale/secure/__init__.py
COPY grid/vpn/secure/api.py /tailscale/secure/api.py
COPY grid/vpn/secure/base_entrypoint.py /tailscale/secure/base_entrypoint.py

ENV HOSTNAME="node"

CMD ["sh", "-c", "/tailscale/tailscale.sh ${HOSTNAME}"]
