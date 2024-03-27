#!/usr/bin/env bash
apt update && apt install -y nginx
nginx &
/app/rathole client.toml
