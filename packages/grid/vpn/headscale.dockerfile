FROM headscale/headscale:0.14.0-alpine

RUN apk add python3 py3-pip
RUN pip install --upgrade pip

WORKDIR /headscale
COPY ./requirements.txt /headscale/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt

COPY ./headscale.sh /headscale/headscale.sh
COPY ./config.yaml /etc/headscale/config.yaml
COPY ./headscale.py /headscale/headscale.py

ENV NETWORK_NAME="omnet"

CMD ["sh", "-c", "/headscale/headscale.sh ${NETWORK_NAME}"]
