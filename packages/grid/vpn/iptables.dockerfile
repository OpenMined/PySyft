FROM alpine:3.18.3

RUN apk update && apk upgrade --available
RUN apk add --no-cache iptables bind-tools bash curl

COPY iptables.sh /iptables.sh
RUN chmod +x /*.sh
