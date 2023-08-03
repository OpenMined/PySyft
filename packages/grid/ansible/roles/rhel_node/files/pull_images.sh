#!/bin/bash
set -e

IMAGES=(
  "docker.io/openmined/grid-frontend:0.8.2-beta.6"
  "docker.io/openmined/grid-backend:0.8.2-beta.6"
  "docker.io/library/mongo:latest"
  "docker.io/traefik:v2.8.1"
  "docker.io/openmined/grid-node-jupyter:0.8.2-beta.6"
)

for img in "${IMAGES[@]}"; do
  echo "Pulling image $img ..."
  podman pull $img
done

echo "All images pulled successfully!"
