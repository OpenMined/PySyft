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
ENV APP_LOG_LEVEL="info"
COPY --from=build /rathole/target/release/rathole /app/rathole
RUN apt update && apt install -y netcat-openbsd vim
WORKDIR /app
COPY ./start.sh /app/start.sh
COPY ./client.toml /app/client.toml
COPY ./server.toml /app/server.toml
COPY ./nginx.conf /etc/nginx/conf.d/default.conf
COPY ./requirements.txt /app/requirements.txt
COPY ./server/ /app/server/

RUN pip install --user -r requirements.txt
CMD ["sh", "-c", "/app/start.sh"]
EXPOSE 2333/udp
EXPOSE 2333

# build and run a fake domain to simulate a normal http container service
# docker build -f domain.dockerfile . -t domain
# docker run --name domain1 -it -d -p 8080:8000 domain



# check the web server is running on 8080
# curl localhost:8080

# build and run the rathole container
# docker build -f rathole.dockerfile . -t rathole

# run the rathole server
# docker run --add-host host.docker.internal:host-gateway --name rathole-server -it -p 8001:8001 -p 8002:8002 -p 2333:2333 -e MODE=server rathole

# check nothing is on port 8001 yet
# curl localhost:8001

# run the rathole client
# docker run --add-host host.docker.internal:host-gateway --name rathole-client -it -e MODE=client rathole

# try port 8001 now
# curl localhost:8001

# add another client and edit the server.toml and client.toml for port 8002


