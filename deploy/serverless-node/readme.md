Before running `terrafrom apply` we would have to create a pair of keys which can be used to ssh into our ec2 instance.
Use the following command to create the key pair:
```shell script
ssh-keygen -b 2048 -t rsa -m PEM -f ec2_efs_key
```
The above command creates two files named `ec2_efs_key`(private key) and `ec2_efs_key.pub`(public key).

Now while running `terraform apply` we would have to enter the above private key file name when prompted to enter a value for `var.key_name`.