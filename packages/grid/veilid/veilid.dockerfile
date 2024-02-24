ARG VEILID_VERSION="0.2.5"

# ======== [Stage 1] Build Veilid Server ========== #
# TODO: Switch from building the packages to using the pre-built packages
# from debian or rpm. This will reduce the build time and the size of the
# final image.
FROM rust as build
ARG VEILID_VERSION
RUN apt update && apt install -y git
RUN git clone -b v${VEILID_VERSION} https://gitlab.com/veilid/veilid
WORKDIR /veilid
RUN bash -c "source scripts/earthly/install_capnproto.sh"
RUN bash -c "source scripts/earthly/install_protoc.sh"
RUN cd veilid-server && cargo build --release -p veilid-server

# ========== [Stage 2] Dependency Install ========== #

FROM python:3.11-bookworm
ARG VEILID_VERSION
COPY --from=build /veilid/target/release/veilid-server /veilid/veilid-server
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt && \
    pip install veilid==${VEILID_VERSION}

COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh
COPY ./veilid.py /app/veilid.py
COPY ./veilid-server.conf /veilid

# ========== [Final] Start Veilid Server and Python Web Server ========== #

CMD ["sh", "-c", "/app/start.sh"]
EXPOSE 5959/udp
EXPOSE 5959
EXPOSE 4000
RUN apt update && apt install netcat-openbsd

# docker build -f veilid.dockerfile . -t veilid
# docker run -it -p 4000:4000 -p 5959:5959 -p 5959:5959/udp veilid
# /root/.local/share/veilid