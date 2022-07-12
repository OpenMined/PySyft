passes=$(cat IPLogin.csv | cut -d "," -f 3)
domain_ips=$(cat IPLogin.csv | cut -d "," -f 1)
usn=$(cat IPLogin.csv | cut -d "," -f 2)

for line in $(cat IPLogin.csv)
do
	ip=$(echo $line | cut -d "," -f 1)
	usn=$(echo $line | cut -d "," -f 2)
	pass=$(echo $line | cut -d "," -f 3)

	echo $ip, $usn, $pass
	# TODO: Replace azureuser to om and use its proper password
	sshpass -p $pass ssh $usn@$ip $(cat update_syft.sh)
	exit
	break
done

