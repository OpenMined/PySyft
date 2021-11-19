#!/bin/sh

# if we have a custom dev mode rootCA.pem it will get loaded
update-ca-certificates

# iptables --list
# block all traffic but port 80 on the tailscale0 interface
# iptables -A INPUT -i tailscale0 -p tcp --dport 80 -j ACCEPT
# iptables -A INPUT -i tailscale0 -p tcp -j REJECT
# iptables -A OUTPUT -p tcp --destination-port 4000 -j DROP

iptables -A INPUT -i tailscale0 -p tcp --destination-port 4000 -j REJECT
iptables -A INPUT -i tailscale0 -p tcp --destination-port 8001 -j REJECT
iptables -A INPUT -i tailscale0 -p tcp --destination-port 8011 -j REJECT
iptables -A INPUT -i tailscale0 -p tcp --destination-port 8080 -j REJECT

export PATH="/root/.local/bin:${PATH}"
export FLASK_APP=tailscale
export HOSTNAME="${1}"
flask run -p 4000 --host=0.0.0.0&
tailscaled
