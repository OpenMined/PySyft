FROM headscale/headscale:0.14.0-alpine

RUN --mount=type=cache,target=/var/cache/apk \
  apk -U upgrade || true; \
  apk fix || true; \
  apk add --no-cache python3 py3-pip curl bash || true

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

ENV NETWORK_NAME="omnet"

CMD ["sh", "-c", "/headscale/headscale.sh ${NETWORK_NAME}"]
