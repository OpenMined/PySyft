# third party
from PyInquirer import prompt
import boto3
import click

# grid relative
from ...utils import Config
from ...utils import styles


def get_vpc_config() -> Config:
    """Assists the user in entering configuration related to VPC.

    :return:
    """

    region = prompt(
        [
            {
                "type": "list",
                "name": "region",
                "message": "Please select your desired AWS region",
                "default": "us-east-1",
                "choices": [
                    region["RegionName"]
                    for region in boto3.client("ec2").describe_regions()["Regions"]
                ],
            }
        ],
        style=styles.second,
    )["region"]

    av_zones = prompt(
        [
            {
                "type": "checkbox",
                "name": "av_zones",
                "message": "Please select at least two availability zones. (Not sure? Select the first two)",
                "choices": [
                    {"name": zone["ZoneName"]}
                    for zone in boto3.client(
                        "ec2", region_name=region
                    ).describe_availability_zones(
                        Filters=[{"Name": "region-name", "Values": [region]}]
                    )[
                        "AvailabilityZones"
                    ]
                ],
            }
        ],
        style=styles.second,
    )["av_zones"]

    return Config(region=region, av_zones=av_zones)


instance_type_filters = {
    "Accelerated Computing Instances(GPU)": [
        {"Name": "instance-type", "Values": ["p*"]}
    ],
    "Compute Optimized Instances": [{"Name": "instance-type", "Values": ["c5*"]}],
    "General Purpose Instances": [{"Name": "instance-type", "Values": ["t2*"]}],
    "Free Tier Instances": [{"Name": "free-tier-eligible", "Values": ["true"]}],
}


def get_all_instance_types_by_filter(client, filter):
    response = client.describe_instance_types(Filters=filter)
    instances = response["InstanceTypes"]

    while "NextToken" in response.keys():
        response = client.describe_instance_types(
            Filters=filter, NextToken=response["NextToken"]
        )
        instances += response["InstanceTypes"]
    return instances


def get_all_instance_names_by_filter(client, filter):
    instances = get_all_instance_types_by_filter(client, filter)
    instance_names = [i["InstanceType"] for i in instances]
    return instance_names


def get_all_instance_types(region):
    client = boto3.client("ec2", region_name=region)
    instance_types = {
        category: get_all_instance_names_by_filter(client, filter)
        for category, filter in instance_type_filters.items()
    }
    return instance_types


def get_instance_type(region):
    # Choose instance category
    client = boto3.client("ec2", region_name=region)
    instance_category = prompt(
        [
            {
                "type": "list",
                "name": "instanceCategory",
                "message": "Please select an AWS instance category.",
                "default": "us-east-1",
                "choices": instance_type_filters.keys(),
            }
        ],
        style=styles.second,
    )["instanceCategory"]

    filter = instance_type_filters[instance_category]
    instances = get_all_instance_types_by_filter(client, filter)

    if len(instances) == 0:
        raise Exception(
            "This region has no instances belonging to your chosen instance category. \nPlease choose a different pair of region and instance category"
        )

    # Sort instances
    sorted_instances = (
        sorted(instances, key=lambda i: i["GpuInfo"]["TotalGpuMemoryInMiB"])
        if instance_category == "Accelerated Computing Instances(GPU)"
        else sorted(instances, key=lambda i: i["VCpuInfo"]["DefaultVCpus"])
    )

    to_GB = lambda x: f"{round(x / 1024, 3)} GB"
    log = lambda name, value: f"{name}: {value}"

    def parse_instance(i):
        s = [" " for i in range(500)]
        s[:30] = log("Instance", i["InstanceType"])
        s[30:60] = log("Memory", to_GB(i["MemoryInfo"]["SizeInMiB"]))
        s[60:80] = log("CPUs", i["VCpuInfo"]["DefaultVCpus"])
        gpu_info = i.get("GpuInfo", None)
        if gpu_info:
            for i, gpu in enumerate(gpu_info["Gpus"]):
                offset = 80 + i * 60
                s[offset : offset + 10] = "GPU :-"
                s[offset + 10 : offset + 25] = gpu["Manufacturer"] + " " + gpu["Name"]
                s[offset + 25 : offset + 45] = "| " + log(
                    "Memory", to_GB(gpu["MemoryInfo"]["SizeInMiB"])
                )
                s[offset + 45 : offset + 60] = "| " + log("Count", gpu["Count"])
            s[80 + (i + 1) * 60 : 80 + (i + 1) * 60 + 90] = log(
                "Total GPU Memory", to_GB(gpu_info["TotalGpuMemoryInMiB"])
            )
        return "".join(s).rstrip()

    # dictionary of parsed instances
    parsed_instances = {parse_instance(i): i for i in sorted_instances}

    # Selct an instance
    instance = prompt(
        [
            {
                "type": "list",
                "name": "instance",
                "message": "Please select your desired AWS instance.",
                "choices": parsed_instances.keys(),
            }
        ],
        style=styles.second,
    )["instance"]

    return Config(InstanceType=parsed_instances[instance]["InstanceType"])


def get_vpc_ip_config() -> Config:
    """Assists the user in entering configuration related to IP address of
    VPC."""

    cidr_blocks = prompt(
        [
            {
                "type": "input",
                "name": "vpc_cidr_block",
                "message": "Please provide VPC cidr block",
                "default": "10.0.0.0/16",
            },
            {
                "type": "input",
                "name": "subnet_cidr_block",
                "message": "Please provide Subnet cidr block",
                "default": "10.0.0.0/24",
            },
        ],
        style=styles.second,
    )

    return Config(
        vpc_cidr_block=cidr_blocks["vpc_cidr_block"],
        subnet_cidr_block=cidr_blocks["subnet_cidr_block"],
    )


def get_db_config() -> Config:
    """Assists the user in entering configuration related the database.

    :return:
    """
    username = prompt(
        [
            {
                "type": "input",
                "name": "username",
                "message": "Please set a username for your Database",
                "validate": lambda x: True
                if len(x) > 4
                else "Username length should be at least 4 characters",
            }
        ],
        style=styles.second,
    )["username"]

    def get_password(msg, validate):
        return prompt(
            [
                {
                    "type": "password",
                    "name": "password",
                    "message": msg,
                    "validate": validate,
                }
            ],
            style=styles.second,
        )["password"]

    password = get_password(
        msg="Enter a password for your Database (length > 8)",
        validate=lambda x: True
        if len(x) > 8
        else "Password length should be greater than 8 characters",
    )
    re_password = get_password(
        msg="Enter the password again",
        validate=lambda x: True
        if x == password
        else "The passwords do not match. Please enter again",
    )

    return Config(username=username, password=password)


aws_lambda_vpc_execution_role_policy = """{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "ec2:CreateNetworkInterface",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DeleteNetworkInterface"
                    ],
                    "Resource": "*"
                }
            ]
        }"""

cloud_watch_logs_full_access_policy = """{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": [
                        "logs:*"
                    ],
                    "Effect": "Allow",
                    "Resource": "*"
                }
            ]
        }
        """

amazon_rds_data_full_access_policy = """{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "SecretsManagerDbCredentialsAccess",
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:PutResourcePolicy",
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:DeleteSecret",
                    "secretsmanager:DescribeSecret",
                    "secretsmanager:TagResource"
                ],
                "Resource": "*"
            },
            {
                "Sid": "RDSDataServiceAccess",
                "Effect": "Allow",
                "Action": [
                    "dbqms:CreateFavoriteQuery",
                    "dbqms:DescribeFavoriteQueries",
                    "dbqms:UpdateFavoriteQuery",
                    "dbqms:DeleteFavoriteQueries",
                    "dbqms:GetQueryString",
                    "dbqms:CreateQueryHistory",
                    "dbqms:DescribeQueryHistory",
                    "dbqms:UpdateQueryHistory",
                    "dbqms:DeleteQueryHistory",
                    "rds-data:ExecuteSql",
                    "rds-data:ExecuteStatement",
                    "rds-data:BatchExecuteStatement",
                    "rds-data:BeginTransaction",
                    "rds-data:CommitTransaction",
                    "rds-data:RollbackTransaction",
                    "secretsmanager:CreateSecret",
                    "secretsmanager:ListSecrets",
                    "secretsmanager:GetRandomPassword",
                    "tag:GetResources"
                ],
                "Resource": "*"
            }
        ]
     }
     """
