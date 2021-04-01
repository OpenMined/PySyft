- Download and install `terraform` from https://www.terraform.io/.
- `cd` to this directory.
- Run `terraform init`.
- Create an AWS account.
- Login to AWS console. Go to `My Security Credentials` and create a new pair of `Access keys`.
- `cd ~ && mkdir .aws`.
- `touch ~/.aws/credentials` to create a file named `credentials` inside `~/.aws`.
- Copy and paste your credentials from AWS console into `~/.aws/credentials`
```
[default]
aws_access_key_id=AKIAIOSFODNN7EXAMPLE
aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```
- cd to this directory and run `terraform apply`.
- To use a different region other than the default one use `terraform apply -var 'region=<region-name>'`. Example `terraform apply -var 'region=us-east-1'`.
- Upon successful creation of resources, above command will output an ip address to the hosted apache server.
- To terminate the resources run `terraform destroy` in this directory.
