FROM headscale/headscale:latest

RUN apt-get update && apt-get install wireguard-tools python3 python3-pip -y

WORKDIR /headscale
COPY ./requirements.txt /headscale/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt

COPY ./headscale.sh /headscale/headscale.sh
COPY ./config.yaml /headscale/config.yaml
COPY ./headscale.py /headscale/headscale.py
COPY ./secure/__init__.py /headscale/secure/__init__.py
COPY ./secure/api.py /headscale/secure/api.py
COPY ./secure/base_entrypoint.py /headscale/secure/base_entrypoint.py

ENV NETWORK_NAME="omnet"

CMD ["sh", "-c", "/headscale/headscale.sh ${NETWORK_NAME}"]
