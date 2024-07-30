ARG RATHOLE_VERSION="0.5.0"
ARG PYTHON_VERSION="3.12"
ARG RUST_VERSION="1.79"

FROM rust:${RUST_VERSION} as build
ARG RATHOLE_VERSION
ARG FEATURES
RUN apt update && apt install -y git
RUN git clone -b v${RATHOLE_VERSION} https://github.com/rapiz1/rathole

WORKDIR /rathole
RUN cargo build --locked --release --features ${FEATURES:-default}

FROM python:${PYTHON_VERSION}-slim-bookworm
ARG RATHOLE_VERSION
ENV MODE="client"
ENV LOG_LEVEL="info"
RUN apt update && apt install -y netcat-openbsd vim rsync
COPY --from=build /rathole/target/release/rathole /app/rathole

WORKDIR /app
COPY ./start.sh /app/start.sh

EXPOSE 2333/udp
EXPOSE 2333

CMD ["sh", "-c", "/app/start.sh"]
