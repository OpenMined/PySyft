#!/bin/bash

REGISTRY=${1:-"k3d-registry.localhost:5800"}
TAG=${2:-"latest"}

docker image build -f ./packages/grid/backend/backend.dockerfile --target backend -t $REGISTRY/openmined/syft-backend:$TAG ./packages
docker image build -f ./packages/grid/frontend/frontend.dockerfile --target syft-ui-development -t $REGISTRY/openmined/syft-frontend:$TAG ./packages/grid/frontend
docker image build -f ./packages/grid/seaweedfs/seaweedfs.dockerfile -t $REGISTRY/openmined/syft-seaweedfs:$TAG ./packages/grid/seaweedfs
