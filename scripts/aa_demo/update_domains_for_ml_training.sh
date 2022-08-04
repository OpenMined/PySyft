#!/bin/bash

passes=$(cat test.csv | cut -d "," -f 3)
domain_ips=$(cat test.csv | cut -d "," -f 1)
usn=$(cat test.csv | cut -d "," -f 2)

for line in $(cat test.csv)
do
	ip=$(echo $line | cut -d "," -f 1)
	usn=$(echo $line | cut -d "," -f 2)
	pass=$(echo $line | cut -d "," -f 3)
	dataset_url=$(echo $line | cut -d "," -f 4)

	COMMAND="cd PySyft &&sudo git restore scripts && sudo git stash && \
			sudo git fetch origin model_training_tests && \
			sudo  git pull origin model_training_tests && \
			sudo git stash apply stash@{0} && \
			sudo  chmod +x scripts/aa_demo/update_domain.sh &&\
			  ./scripts/aa_demo/update_domain.sh $ip $dataset_url"
	echo $ip, $usn, $pass
	# TODO: Replace azureuser to om and use its proper password
	# sshpass -p $pass ssh $usn@$ip $(cat update_syft.sh)
	sshpass  -p $pass ssh $usn@$ip -o StrictHostKeyChecking=no "sudo runuser -l om -c '${COMMAND}'"

done

