#!/bin/ash

# if we have a custom dev mode rootCA.pem it will get loaded
update-ca-certificates

# iptables --list
# block all traffic but port 80 on the tailscale0 interface
# iptables -A INPUT -i tailscale0 -p tcp --dport 80 -j ACCEPT
# iptables -A INPUT -i tailscale0 -p tcp -j REJECT
# iptables -A OUTPUT -p tcp --destination-port 4000 -j DROP

# we use 81 when SSL is enabled to redirect external port 80 traffic to SSL
# however the VPN can't use SSL because the certs cant be for IPs and the traffic
# is encrypted anyway so we dont need it
iptables -A INPUT -i tailscale0 -p tcp --destination-port 81 -j REJECT
# additionally if SSL is enabled we might be using 443+ in testing to provide
# multiple different stacks, 443, 444, 446 etc however this should be blocked
# over the VPN so we dont accidentally use it somehow
iptables -A INPUT -i tailscale0 -p tcp --destination-port 443:450 -j REJECT
iptables -A INPUT -i tailscale0 -p tcp --destination-port 4000 -j REJECT
iptables -A INPUT -i tailscale0 -p tcp --destination-port 8001 -j REJECT
iptables -A INPUT -i tailscale0 -p tcp --destination-port 8011 -j REJECT
iptables -A INPUT -i tailscale0 -p tcp --destination-port 8080 -j REJECT

# allow k8s pods to use tailscale as a Gateway NAT
iptables -t nat -A POSTROUTING -o tailscale0 -j MASQUERADE
# on each container you will need to set the IP of the tailscale container
# as the gateway for a new route
# ip route add 100.64.0.0/24 via $TAILSCALE_CONTAINER_IP dev eth0 onlink
# see tailscale-gateway.sh

export PATH="/root/.local/bin:${PATH}"
export FLASK_APP=tailscale
export HOSTNAME="${1}"
flask run -p 4000 --host=0.0.0.0&
tailscaled -port 41641
