# Ansible playbook to deploy PyGrid components (for Debian based systems)

## Install ansible

Before running ansible, it needs to be installed. One can find more
information about installation to your linux distribution [here](http://docs.ansible.com/ansible/intro_installation.html).


First, you need setup a [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html). Follow the commands:

```
virtualenv venv
source venv/bin/activate
```

Installing Ansible with pip:

```
$ pip install ansible
```

### Editing **/etc/ansible/ansible.cfg**

#### To not ask ssh key host checking:

SSH always ask if you sure in establish a ssh connection in your first time that you try to connect in a machine. For jump this, add the following in "defaults" section:

```
host_key_checking = False
```

#### Enabling privilege escalations:

This playbook tasks need superuser privilege to run correctly. As above, you can add this in "defaults" section:

```
become = True
```


## Add keys in machine that will be managed

To Ansible install and configure machine, it's necessary that your key is added in
**.ssh/authorized_keys** in machine that will be managed. To do this, run the command
below:

```
$ ssh-copy-id -i ~/<route_ssh_public_key> <user>@<ip_or_hostname>
```

## Add hosts to the inventory

Before anything, add the hosts that will receive the containers to inventory file like:

```
[hosts]
0.0.0.0 #host1
127.0.0.1 #host2
```

## Run the playbook

Run the playbook using the command below:

```
$ ansible-playbook -i inventory pygrid_deploy.yaml
```

## Destroy containers

To destroy the containers created by deploy playbook, run the following command:

```
$ ansible-playbook -i inventory destroy.yaml
```

#### WARNING:
PyGrid nodes require PyGrid gateway and Redis service to be up and running, so deploy them first (always put their tasks before in the playbook)

