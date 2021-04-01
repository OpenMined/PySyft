# Deployment using PyGrid CLI

OpenMined PyGrid CLI provide the necessary support for Infrastructure and Deployment management of various PyGrid apps (Domains and Networks) to the main cloud providers ([AWS](https://aws.amazon.com/), [GCP](https://cloud.google.com/), [Azure](https://azure.microsoft.com/)) through the usage of [Terraform](https://www.terraform.io/) and [TerraScript](https://github.com/mjuenema/python-terrascript) integrated within the CLI.

## Installing Terraform

Check Instructions here:
- https://www.terraform.io/downloads.html
- https://learn.hashicorp.com/tutorials/terraform/install-cli

## Cloud Providers

PyGrid CLI provides full support for the following cloud providers

- **Amazon Web Services ([AWS](https://aws.amazon.com/))**

- **Google Cloud Platform ([GCP](https://cloud.google.com/))**

- **Microsoft Cloud Computing Services ([Azure](https://azure.microsoft.com/en-us/)**)

### Getting Cloud Credentials

#### AWS credentials

You need to obtain your AWS IAM user credentials (which has programmatic access enabled) and add them to the following files in the home directory.

In `~/.aws/credentials.json`
```json
{
    "aws_access_key_id": "XXXXXXXXXXXXXXXXXXXX",
    "aws_secret_access_key": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
}
```
& in `~/.aws/credentials`
```shell script
[default]
aws_access_key_id=XXXXXXXXXXXXXX
aws_secret_access_key=XXXXXXXXXXXXXXXXXXXX
```

- [Instructions to create new IAM user with programmatic access](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html)
- [Instructions for finding your keys](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html#access-keys-and-secret-access-keys)

#### GCP credentials

You will need to use a [GCP service account](https://cloud.google.com/docs/authentication/getting-started) to provide authentication and obtain required credentials

[Instructions for adding credentials](https://registry.terraform.io/providers/hashicorp/google/latest/docs/guides/getting_started#adding-credentials)

Or through using [GCloud CLI](https://cloud.google.com/sdk/docs/install) and loggining to your GCP provider through this command:

```shell
gcloud auth application-default login
```

#### AZURE credentials

You need these keys

```shell
subscription_id = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
client_id = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
client_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
tenant_id = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

and provide them to PyGrid CLI as AZURE credentials

- [Creating a Service Principal to get the client_id, client_secret, and tenant_id fields](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret#creating-a-service-principal)
- [Instructions for finding values for subscription_id](https://blogs.msdn.microsoft.com/mschray/2015/05/13/getting-your-azure-guid-subscription-id/)
- [Instructions for finding values for tenant_id, application_id, and key](https://docs.microsoft.com/en-ca/azure/active-directory/develop/howto-create-service-principal-portal)


## Installing the CLI

### Using pip (recommended)

```shell
pip install pygrid-cli
```

### Manual install

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

## Deployment without using the CLI

The CLI allows a good amount of flexibility to meet your deployment requirements. But if you need to do a custom/manual deployment, follow these steps.
- Deploy a VPC, domain server, subnets, route tables, network gateways, database and connect them altogether.

After deploying the domain server, you would need to SSH into it and perform these tasks.
- Install `Terraform`.
- Clone `PyGrid`.
- Set up environment variables (Eg. for amazon `DATABASE_URL`, `CLOUD_PROVIDER`, `REGION`, `VPC_ID`, `PUBLIC_SUBNET_ID`, `PRIVATE_SUBNET_ID`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- Start the domain/network app using the `run.sh` file (found in respective app folders).
```shell script
nohup ./run.sh --port 5000 --host 0.0.0.0
```
For better understanding of steps involved in setting up the domain, you can refer to our post-deployment scripts in the `write_domain_exec_script` function of the respective cloud provider classes (`AWS_Serverfull`, `Azure`, `GCP`) found in `apps/domain/src/main/core/infrastructure/providers/`.


## Serverless deployment

Serverless deployment has risen in popularity in recent years for a variety of reasons:

- Ease of deployment: You no longer need to be concerned about process management or configuring a network to accept appropriate incoming connections.
- Auto-scaling: Serverless functions scale implicitly because of the fact that they are not running on a continuous process, and are instead “instantiated on-demand”.
- Inexpensive: Because you’re not paying for a continuously running process, you’re only paying when the function is called, which is significantly cheaper than paying for a constantly running server.

As such, PyGrid Domains and Networks, which are simply message forwarding services and API’s with an attached SQL database, are great candidates for serverless deployment. Workers, on the other hand, due to their specific hardware requirements necessary for computation, should always be deployed in a “serverfull” (traditional) configuration. While this may sound expensive, if a Worker is deployed to a cloud provider then the data scientist has the ability to turn on and off the instance (the Worker) at their discretion.

*Note: The PyGrid team wrote an initial serverless deployment setup for Domains and Networks. But, currently this functionality does not exist in a completed form. The current constraints are actually due to the fact that `PySyft` has a hard dependency on PyTorch at the moment. Currently it’s not possible to utilize only the class structure required for parsing messages from PySyft without having to install PyTorch as well. Serverless functions often come with file size constraints, making this impossible or unrealistic. It’s the PyGrid team’s hope that this dependency is later marked as optional, allowing the team to simply utilize the PySyft class structure inside of PyGrid, without depending on any deep learning framework functionality.*

If serverless deployment becomes a reality for Domains and Networks, then it is our suggestion that you deploy Domains and Networks in a serverless configuration. This is the optimal method of deployment given cost, scaling concerns, and maintenance effort. Using this method, it could mean that a data owner would be able to host their private data in a PyGrid serverless deployment, and never need to be concerned with ongoing scale or maintenance of their deployment - truly a “set it and forget it” situation.
