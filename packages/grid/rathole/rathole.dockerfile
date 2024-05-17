ARG RATHOLE_VERSION="0.5.0"
ARG PYTHON_VERSION="3.12"

FROM rust as build
ARG RATHOLE_VERSION
ARG FEATURES
RUN apt update && apt install -y git
RUN git clone -b v${RATHOLE_VERSION} https://github.com/rapiz1/rathole

WORKDIR /rathole
RUN cargo build --locked --release --features ${FEATURES:-default}

FROM python:${PYTHON_VERSION}-bookworm
ARG RATHOLE_VERSION
ENV MODE="client"
COPY --from=build /rathole/target/release/rathole /app/rathole
RUN apt update && apt install -y netcat-openbsd vim
WORKDIR /app
COPY ./start-client.sh /app/start-client.sh
COPY ./start-server.sh /app/start-server.sh
COPY ./client.toml /app/client.toml
COPY ./server.toml /app/server.toml
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

CMD ["sh", "-c", "/app/start-$MODE.sh"]
EXPOSE 2333/udp
EXPOSE 2333

# build and run a fake domain to simulate a normal http container service
# docker build -f domain.dockerfile . -t domain
# docker run -it -d -p 8080:8000 domain

# check the web server is running on 8080
# curl localhost:8080

# build and run the rathole container
# docker build -f rathole.dockerfile . -t rathole

# build nginx container
# docker build -f nginx.dockerfile . -t proxy
# docker run -it -p 9090:9090 proxy bash

# run the rathole server
# docker run -it -p 8001:8001 -p 8002:8002 -p 2333:2333 -e MODE=server rathole

# check nothing is on port 8001 yet
# curl localhost:8001

# run the rathole client
# docker run -it -e MODE=client rathole

# try port 8001 now
# curl localhost:8001

# add another client and edit the server.toml and client.toml for port 8002




# docker build -f domain.dockerfile . -t domain
# docker build -f rathole.dockerfile . -t rathole
# docker build -f nginx.dockerfile . -t proxy

# docker run -it -d -p 8080:8000 domain
# docker run -it -p 9090:9090 proxy bash
# docker run -it -p 8001:8001 -p 8002:8002 -p 2333:2333 -e MODE=server rathole
# docker run -it -e MODE=client rathole