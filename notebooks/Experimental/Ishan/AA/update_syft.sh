DOMAIN_NAME="$(sudo docker ps --format '{{.Names}}' | grep "celery" |rev | cut -c 16- | rev)"
echo "Domain Name: ${DOMAIN_NAME}"
echo "Nuking  Domain ..  >:)";
hagrid land all;
echo "Launching Domain .. hahaha >:)";
hagrid launch ${DOMAIN_NAME} to docker:80  --tail=false --dev
echo "finished"

