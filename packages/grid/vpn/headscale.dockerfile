ARG PYTHON_VERSION='3.11'

FROM python:3.11-slim as headscale

# set UTC timezone
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /root/.local

RUN apt-get update && apt-get upgrade -y
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
  DEBIAN_FRONTEND=noninteractive \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  curl procps && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ENV WAITFORIT_VERSION="v2.4.1"
RUN curl -o /usr/local/bin/waitforit -sSL https://github.com/maxcnunes/waitforit/releases/download/$WAITFORIT_VERSION/waitforit-linux_amd64 && \
  chmod +x /usr/local/bin/waitforit

ENV HEADSCALE_VERSION="0.22.3"
RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) != "x86_64" ]; then \
  curl -o /bin/headscale -sSL https://github.com/juanfont/headscale/releases/download/v${HEADSCALE_VERSION}/headscale_${HEADSCALE_VERSION}_linux_arm64; \
  fi

RUN --mount=type=cache,target=/root/.cache if [ $(uname -m) == "x86_64" ]; then \
  curl -o /bin/headscale -sSL https://github.com/juanfont/headscale/releases/download/v${HEADSCALE_VERSION}/headscale_${HEADSCALE_VERSION}_linux_amd64; \
  fi

RUN chmod +x /bin/headscale

RUN mkdir -p /var/run/headscale

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
