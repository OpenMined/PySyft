ARG RATHOLE_VERSION="0.5.0"
ARG PYTHON_VERSION="3.12"

FROM rapiz1/rathole:v${RATHOLE_VERSION} as build

FROM python:${PYTHON_VERSION}-bookworm
ARG RATHOLE_VERSION
ENV MODE="client"
RUN apt update && apt install -y netcat-openbsd vim
COPY --from=build /app/rathole   /app/rathole

WORKDIR /app
COPY ./start.sh /app/start.sh

EXPOSE 2333/udp
EXPOSE 2333

CMD ["sh", "-c", "/app/start.sh"]


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


