#!/bin/sh

# block all traffic but port 80 on the tailscale0 interface
# iptables -A INPUT -i tailscale0 -p tcp --dport 80 -j ACCEPT
# iptables -A INPUT -i tailscale0 -p tcp -j REJECT

export PATH="/root/.local/bin:${PATH}"
export FLASK_APP=tailscale
export HOSTNAME="${1}"
flask run -p 4000 --host=0.0.0.0&
tailscaled
