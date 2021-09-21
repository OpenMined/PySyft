#!/bin/bash
HAGRID_VERSION=$(python3 hagrid/__init__.py)
docker buildx build -f hagrid.dockerfile -t openmined/hagrid:"${HAGRID_VERSION}" -t openmined/hagrid:latest .
