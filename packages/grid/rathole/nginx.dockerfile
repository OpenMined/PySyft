ARG PYTHON_VERSION="3.12"
FROM python:${PYTHON_VERSION}-bookworm
RUN apt update && apt install -y netcat-openbsd vim
RUN apt update && apt install -y nginx
COPY ./proxy.conf /etc/nginx/conf.d/default.conf
CMD ["nginx"]
