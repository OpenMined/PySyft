FROM rust as build
RUN apt update && apt install -y git
RUN git clone -b v0.2.5 https://gitlab.com/veilid/veilid
WORKDIR /veilid
RUN bash -c "source scripts/earthly/install_capnproto.sh"
RUN bash -c "source scripts/earthly/install_protoc.sh"
RUN cd veilid-server && cargo build --release -p veilid-server -p veilid-cli -p veilid-tools -p veilid-core

RUN cp /veilid/target/release/veilid-server /app

FROM python:3.11-bookworm
COPY --from=build /veilid/target/release/veilid-server /veilid/veilid-server
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt
COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh
COPY ./veilid.py /app/veilid.py
COPY ./veilid-server.conf /veilid
CMD ["sh", "-c", "/app/start.sh"]
EXPOSE 5959/udp
EXPOSE 5959
EXPOSE 4000
RUN apt update && apt install netcat-openbsd
# docker build -f veilid.dockerfile . -t veilid
# docker run -it -p 4000:4000 -p 5959:5959 -p 5959:5959/udp veilid
# /root/.local/share/veilid