### Installing and Configuring Terraform
- Download and install `terraform` from https://www.terraform.io/.
- `cd` to this directory.
- Login to AWS console. Go to `My Security Credentials` and create a new pair of `Access keys`.
- `cd ~ && mkdir .aws`.
- `touch ~/.aws/credentials` to create a file named `credentials` inside `~/.aws`.
- Copy and paste your credentials from AWS console into `~/.aws/credentials`
```
[default]
aws_access_key_id=AKIAIOSFODNN7EXAMPLE
aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

### Deploying resources
- `cd` into the directory of the PyGrid component you wish to deploy.
- Run `terraform init`.
- Run `terraform apply`.
- To use a different region other than the default one use `terraform apply -var 'region=<region-name>'`. Example `terraform apply -var 'region=us-east-1'`.
- Upon successful creation of resources, above command will output an endpoint to access the deployment.
- To terminate the deployment run `terraform destroy`.
