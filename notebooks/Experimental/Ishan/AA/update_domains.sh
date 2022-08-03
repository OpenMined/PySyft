passes=$(cat test.csv | cut -d "," -f 3)
domain_ips=$(cat test.csv | cut -d "," -f 1)
usn=$(cat test.csv | cut -d "," -f 2)

for line in $(cat test.csv)
do
	ip=$(echo $line | cut -d "," -f 1)
	usn=$(echo $line | cut -d "," -f 2)
	pass=$(echo $line | cut -d "," -f 3)

	COMMAND="cd PySyft &&git stash && \
			sudo git fetch origin model_training_tests && \
			 git pull origin model_training_tests && \
			  git stash apply stash@{0} && \
			  ./scripts/aa_demo/update_domain.sh"
	echo $ip, $usn, $pass
	# TODO: Replace azureuser to om and use its proper password
	# sshpass -p $pass ssh $usn@$ip $(cat update_syft.sh)
	sshpass -p $pass ssh $usn@$ip  "sudo runuser -l om -c '${COMMAND} && ls '"
	exit
	break
done

