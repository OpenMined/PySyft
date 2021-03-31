# Deployment & CLI

OpenMined PyGrid CLI provide the necessary support for Infrastructure and Deployment management of various PyGrid apps including (Domains, Networks, and Workers) to the main cloud providers ([AWS](https://aws.amazon.com/), [GCP](https://cloud.google.com/), [Azure](https://azure.microsoft.com/)) through the usage of [Terraform](https://www.terraform.io/) and [TerraScript](https://github.com/mjuenema/python-terrascript) integrated within the CLI.

## Installing Terraform

Check Instructions here:
- https://www.terraform.io/downloads.html
- https://learn.hashicorp.com/tutorials/terraform/install-cli


## Installing the CLI

To get started, install the CLI first through thess commands:

Clone PyGrid Repo:

```shell
git clone https://github.com/OpenMined/PyGrid.git
```

Then install the python package through pip

```shell
pip install -e .
```

## CLI Instructions

Using the PyGrid CLI is very simple through this command `pygrid deploy` then following the instruction to successfully deploying a specific pygrid app, described in the commands below

### List of CLI Commands

#### Deploy a Domain to AWS

```shell
pygrid deploy --provider aws --app domain
```

#### Deploy a Network to Azure

```shell
pygrid deploy --provider azure --app network
```

CLI Instructions Example

```shell
$ pygrid deploy                                                                                                                                           [7h 19m]
Welcome to OpenMined PyGrid CLI
Cloud Provider:  (AWS, GCP, AZURE) [AWS]: AWS
PyGrid App:  (Domain, Network) [Domain]: Domain
Please enter path to your  aws credentials json file [/Users/amrmkayid/.aws/credentials.json]:
Will you need to support Websockets? [y/N]: N
How many servers do you wish to deploy? (All are managed under the load balancer) [1]: 1
#1: PyGrid Domain ID: domain-demo
#1: Port number of the socket.io server [5000]:
? Please select your desired AWS region  eu-west-2
? Please select at least two availability zones. (Not sure? Select the first two)  done (3 selections)
? Please select an AWS instance category.  General Purpose Instances
? Please select your desired AWS instance.  Instance: t2.xlarge           Memory: 16.0 GB               CPUs: 4
? Please set a username for your Database  amrmkayid
? Enter a password for your Database (length > 8)  **********
? Enter the password again  **********
Your current configration are:

{
  "pygrid_root_path": "/Users/amrmkayid/.pygrid/cli",
  "output_file": "/Users/amrmkayid/.pygrid/cli/config_2021-03-30_011924.json",
  "provider": "aws",
  "app": {
    "name": "domain",
    "count": 1
  },
  "root_dir": "/Users/amrmkayid/.pygrid/apps/aws/domain",
  "serverless": false,
  "websockets": false,
  "apps": [
    {
      "id": "domain-demo",
      "port": 5000
    }
  ],
  "vpc": {
    "region": "eu-west-2",
    "av_zones": [
      "eu-west-2a",
      "eu-west-2b",
      "eu-west-2c"
    ],
    "instance_type": {
      "InstanceType": "t2.xlarge"
    }
  }
}

Continue? [y/N]:
```

## Serverfull and Serverless deployment

## Resource Managers (PM2)


## Cloud Providers

PyGrid CLI provide the full suport for the following cloud providers

- **Amazon Web Services ([AWS](https://aws.amazon.com/))**: Both serverfull and serverless deployment

- **Google Cloud Platform ([GCP](https://cloud.google.com/))**: serverfull deployment

- **Microsoft Cloud Computing Services [Azure](https://azure.microsoft.com/en-us/)**: serverfull deployment

### Getting Cloud Credentials

#### AWS credentials

You need to obtain

```json
{
    "aws_access_key_id": "XXXXXXXXXXXXXXXXXXXX",
    "aws_secret_access_key": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
}
```

and store the credentials into this folder `~/.aws/credentials.json` in your home directory

- Instructions for finding your keys are here:
https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html#access-keys-and-secret-access-keys

#### GCP credentials

You will need to use a [GCP service account](https://cloud.google.com/docs/authentication/getting-started) to provide authentication and obtain required credentials

Instructions for adding credentials here:
https://registry.terraform.io/providers/hashicorp/google/latest/docs/guides/getting_started#adding-credentials

Or through using [GCloud CLI](https://cloud.google.com/sdk/docs/install) and loggining to your GCP provider through this command:

```shell
gcloud auth application-default login
```

#### AZURE credentials

You need to these keys

```shell
application_id = "APPLICATION_ID"
subscription_id = "SUBSCRIPTION_ID"
tenant_id = "TENANT_ID"
key = "KEY"
```

and provide them to PyGrid CLI as AZURE credentials


- Instructions for finding values for subscription_id:
https://blogs.msdn.microsoft.com/mschray/2015/05/13/getting-your-azure-guid-subscription-id/

- Instructions for finding values for tenant_id, application_id, key:
https://docs.microsoft.com/en-ca/azure/active-directory/develop/howto-create-service-principal-portal
