#!/bin/bash

RG='dockerswarmtest'
Image='Canonical:0001-com-ubuntu-server-focal:20_04-lts:latest'
VM_NAME='dockerswarm-machine'
Image_size='Standard_D8s_v5'
Location='westus'
Admin='om'

#Create Resource Group
az group create --name $RG --location westus

#Create VM
az vm create \
    --resource-group $RG \
    --name $VM_NAME \
    --image $Image \
    --size $Image_size \
    --count 5 \
    --data-disk-sizes-gb 250 \
    --location $Location \
    --admin-username $Admin \
    --generate-ssh-keys \

# #Open Port 80 on all VMs
 az vm open-port --ids $(az vm list -g $RG --query "[].id" -o tsv) --port 80



# #Get names of all VMs in the Resource Group
 VMs=$(az vm list -g $RG --query "[].name" -o tsv)




# #Loop through all VMs and install Docker
for VM in $VMs
do
    az vm run-command invoke -g $RG -n $VM --command-id RunShellScript --scripts "sudo apt-get update && sudo apt-get install docker.io -y"
done

# #Loop through all Vms and add permission to run docker commands without sudo
for VM in $VMs
do
    az vm run-command invoke -g $RG -n $VM --command-id RunShellScript --scripts "chmod 777 /var/run/docker.sock"  #This is for test and we need to do it right for sec issue
done


# Get the name of the first VM
consul_machine=$(az vm list -g $RG --query "[0].name" -o tsv)
echo $consul_machine
consul_machine_eth0_inet_ip=$(az vm show -g $RG -n $consul_machine --query privateIps -d --out tsv)
echo $consul_machine_eth0_inet_ip

# #Install  consul on the first VM for service discovery
az vm run-command invoke -g $RG -n $consul_machine --command-id RunShellScript --scripts "docker run -d -p 8500:8500 --name=consul progrium/consul -server -bootstrap"


# #Get the name & IP address of the second VM
manager_machine1=$(az vm list -g $RG --query "[1].name" -o tsv)
echo $manager_machine1
manager_machine1_eth0_inet_ip=$(az vm show -g $RG -n $manager_machine1 --query privateIps -d --out tsv)
echo $manager_machine1_eth0_inet_ip

# #Make Swarm Manager on the manager machine 1
az vm run-command invoke -g $RG -n $manager_machine1 --command-id RunShellScript --scripts "docker run -d -p 4000:4000 swarm manage -H :4000 --replication --advertise $manager_machine1_eth0_inet_ip:4000 consul://$consul_machine_eth0_inet_ip:8500"


# #Get the name & IP address of the third VM
manager_machine2=$(az vm list -g $RG --query "[2].name" -o tsv)
echo $manager_machine2
manager_machine2_eth0_inet_ip=$(az vm show -g $RG -n $manager_machine2 --query privateIps -d --out tsv)
echo $manager_machine2_eth0_inet_ip

# #Make Swarm Manager Replica on the manager machine 2
az vm run-command invoke -g $RG -n $manager_machine2 --command-id RunShellScript --scripts "docker run -d -p 4000:4000 swarm manage -H :4000 --replication --advertise $manager_machine2_eth0_inet_ip:4000 consul://$consul_machine_eth0_inet_ip:8500"


#  #Get the name & IP address of the fourth VM
worker_machine1=$(az vm list -g $RG --query "[3].name" -o tsv)
echo $worker_machine1
worker_machine1_eth0_inet_ip=$(az vm show -g $RG -n $worker_machine1 --query privateIps -d --out tsv)
echo $worker_machine1_eth0_inet_ip

#Join the cluster for as Docker Swarm Worker on the worker machine 1
az vm run-command invoke -g $RG -n $worker_machine1 --command-id RunShellScript --scripts "docker run -d swarm join --advertise=$worker_machine1_eth0_inet_ip:2375 consul://$consul_machine_eth0_inet_ip:8500"

#Get the name & IP address of the fifth VM
worker_machine2=$(az vm list -g $RG --query "[4].name" -o tsv)
echo $worker_machine2
worker_machine2_eth0_inet_ip=$(az vm show -g $RG -n $worker_machine2 --query privateIps -d --out tsv)
echo $worker_machine2_eth0_inet_ip

#Join the cluster for as Docker Swarm Worker on the worker machine 2
az vm run-command invoke -g $RG -n $worker_machine2 --command-id RunShellScript --scripts "docker run -d swarm join --advertise=$worker_machine2_eth0_inet_ip:2375 consul://$consul_machine_eth0_inet_ip:8500"

