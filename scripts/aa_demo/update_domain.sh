#!/bin/bash

# $1 domain ip
# $2 dataset url
# install syft in dev mode
pip install -U -e packages/syft

# get domain name
NODE_NAME="$(sudo docker ps --format '{{.Names}}' | grep "celery" |rev | cut -c 16- | rev)"
echo "Domain Name: ${NODE_NAME}"
echo "Nuking  Domain ..  >:)";

# destroy current domain
hagrid land all
echo "Launching Domain .. hahaha >:)";

# re-launch domain
hagrid launch ${NODE_NAME} to docker:80 --dev --build-src="model_training_tests"

# wait for domain to be up
hagrid check --timeout=120

echo "Domain lauch succeeded."
echo "Starting to upload dataset"

# upload dataset
python scripts/aa_demo/upload_dataset.py $1 $2

echo "Upload dataset script complete."
