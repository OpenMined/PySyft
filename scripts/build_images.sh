#!/bin/bash

docker image build -f ./packages/grid/backend/backend.dockerfile --target backend -t syftpilot.azurecr.io/openmined/grid-backend:$1 ./packages
docker image build -f ./packages/grid/frontend/frontend.dockerfile --target grid-ui-development -t syftpilot.azurecr.io/openmined/grid-frontend:$1 ./packages/grid/frontend
docker image build -f ./packages/grid/seaweedfs/seaweedfs.dockerfile --build-arg SEAWEEDFS_VERSION=3.59 -t syftpilot.azurecr.io/openmined/grid-seaweedfs:$1 ./packages/grid/seaweedfs
