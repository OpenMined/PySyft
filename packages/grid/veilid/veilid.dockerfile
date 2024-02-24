ARG VEILID_VERSION="0.2.5"


FROM python:3.11-bookworm
ARG VEILID_VERSION

# ========== [Stage 1] Install Veilid Server ========== #

RUN wget -O- https://packages.veilid.net/gpg/veilid-packages-key.public \
    | gpg --dearmor -o /usr/share/keyrings/veilid-packages-keyring.gpg

RUN ARCH=$(dpkg --print-architecture) && \
    echo "deb [arch=$ARCH signed-by=/usr/share/keyrings/veilid-packages-keyring.gpg] https://packages.veilid.net/apt stable main" \
    > /etc/apt/sources.list.d/veilid.list

RUN apt update && apt install -y veilid-server=${VEILID_VERSION} && apt-get clean


# ========== [Stage 2] Install Dependencies ========== #

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    pip install --user -r requirements.txt && \
    pip install veilid==${VEILID_VERSION}

COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh
COPY ./server /app/server
COPY ./veilid-server.conf /veilid

# ========== [Final] Start Veilid Server and Python Web Server ========== #

CMD ["sh", "-c", "/app/start.sh"]
EXPOSE 5959/udp
EXPOSE 5959
EXPOSE 4000

# RUN apt update && apt install netcat-openbsd
# docker build -f veilid.dockerfile . -t veilid
# docker run -it -p 4000:4000 -p 5959:5959 -p 5959:5959/udp veilid
# /root/.local/share/veilid