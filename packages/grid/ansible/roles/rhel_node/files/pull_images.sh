#!/bin/bash

podman pull docker.io/openmined/grid-frontend:0.8.2-beta.6
podman pull docker.io/openmined/grid-backend:0.8.2-beta.6
podman pull docker.io/library/mongo:latest
podman pulldocker.io/traefik:v2.8.1
podman pull docker.io/openmined/grid-node-jupyter:0.8.2-beta.6