FROM headscale/headscale:0.11

RUN apt-get update && apt-get install wireguard-tools python3 python3-pip -y

WORKDIR /headscale
COPY ./requirements.txt /headscale/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt

COPY ./headscale.sh /headscale/headscale.sh
COPY ./config.yaml /headscale/config.yaml
COPY ./headscale.py /headscale/headscale.py

ENV NETWORK_NAME="omnet"

CMD ["sh", "-c", "/headscale/headscale.sh ${NETWORK_NAME}"]
