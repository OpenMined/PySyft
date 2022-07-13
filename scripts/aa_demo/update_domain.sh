# $1 domain ip
# $2 dataset url
# install syft in dev mode
pip install -e packages/syft

# get domain name
DOMAIN_NAME="$(sudo docker ps --format '{{.Names}}' | grep "celery" |rev | cut -c 16- | rev)"
echo "Domain Name: ${DOMAIN_NAME}"
echo "Nuking  Domain ..  >:)";

# destroy current domain
hagrid land all
echo "Launching Domain .. hahaha >:)";

# re-launch domain
hagrid launch ${DOMAIN_NAME} to docker:80  --tail=false --dev

# wait for domain to be up
hagrid check --wait --silent

echo "Domain lauch succeeded."
echo "Starting to upload dataset"

# upload dataset
python upload_dataset.py $1 $2

echo "Upload dataset script complete."