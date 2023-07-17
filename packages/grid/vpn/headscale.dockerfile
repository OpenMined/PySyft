FROM headscale/headscale:0.22.3

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
  DEBIAN_FRONTEND=noninteractive \
  apt-get update && \
  apt-get install -yqq \
  python3 python3-pip curl procps && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ENV WAITFORIT_VERSION="v2.4.1"
RUN curl -o /usr/local/bin/waitforit -sSL https://github.com/maxcnunes/waitforit/releases/download/$WAITFORIT_VERSION/waitforit-linux_amd64 && \
  chmod +x /usr/local/bin/waitforit

WORKDIR /headscale
COPY ./requirements.txt /headscale/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
  pip install --user -r requirements.txt

COPY ./headscale.sh /headscale/headscale.sh
COPY ./config.yaml /etc/headscale/config.yaml
COPY ./headscale.py /headscale/headscale.py
RUN mkdir -p /headscale/data

ENV NETWORK_NAME="omnet"

CMD ["sh", "-c", "/headscale/headscale.sh ${NETWORK_NAME}"]
