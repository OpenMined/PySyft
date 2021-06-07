# stdlib
import subprocess

# grid relative
from .aws import *
from .utils import *


class AWS_Serverless(AWS):
    def __init__(self, config) -> None:
        """
        config (Config) : Object storing the required configuration for deployment
        """

        super().__init__(config)

        # TODO: Make it beautiful
        try:
            self.python_runtime = config.app.python_runtime
        except AttributeError:
            self.python_runtime = "python3.8"

        # Api gateway
        self.build_api_gateway()

        # Database
        self.build_database()
        self.build_secret_manager()

        # Lambda function associates
        self.build_lambda_role()
        self.build_security_group()
        self.build_lambda_layer()

        # Main lambda function
        self.build_lambda_function()

        # Append outputs
        self.outputs()

    def outputs(self):
        """Add outputs to be returned as a response from the API."""
        self.tfscript += terrascript.Output(
            "api_gateway_endpoint",
            value=var_module(self.api_gateway, "this_apigatewayv2_api_api_endpoint"),
            description=f"PyGrid {self.config.app.name} API endpoint",
        )

    def build_lambda_layer(self):
        """Creates a AWS S3 bucket object and uploads zipped dependencies (of
        the app) to it.

        Then creates a lambda layer, which points to that S3 bucket.
        This lambda layer is later attached to the lambda function
        hosting the app.
        """

        s3_bucket = resource.aws_s3_bucket(
            f"{self.config.app.name}-lambda-layer-bucket",
            bucket=f"pygrid-{self.config.app.name}-lambda-layer-bucket",
            acl="private",
            versioning={"enabled": True},
            tags={"Name": f"pygrid-{self.config.app.name}-s3-bucket"},
        )
        self.tfscript += s3_bucket

        dependencies_zip_path = self.zip_dependencies()

        s3_bucket_object = resource.aws_s3_bucket_object(
            f"pygrid-{self.config.app.name}-lambda-layer",
            bucket=s3_bucket.bucket,
            key=var('filemd5("{}")'.format(dependencies_zip_path)) + ".zip",
            source=dependencies_zip_path,
            depends_on=[f"aws_s3_bucket.{s3_bucket._name}"],
            tags={"Name": f"pygrid-{self.config.app.name}-s3-bucket-object"},
        )
        self.tfscript += s3_bucket_object

        self.lambda_layer = resource.aws_lambda_layer_version(
            f"pygrid-{self.config.app.name}-lambda-layer",
            layer_name=f"pygrid-{self.config.app.name}-dependencies",
            compatible_runtimes=[self.python_runtime],
            s3_bucket=s3_bucket_object.bucket,
            s3_key=s3_bucket_object.key,
            depends_on=[f"aws_s3_bucket_object.{s3_bucket_object._name}"],
        )
        self.tfscript += self.lambda_layer

    def build_api_gateway(self):
        """Builds an API Gateway, which points to the main lambda function."""
        self.api_gateway = Module(
            "api_gateway",
            source="terraform-aws-modules/apigateway-v2/aws",
            name=f"pygrid-{self.config.app.name}-http",
            protocol_type="HTTP",
            create_api_domain_name=False,
            integrations={
                "$default": {"lambda_arn": "${module.lambda.this_lambda_function_arn}"}
            },
            tags={"Name": f"pygrid-{self.config.app.name}-api-gateway-http"},
        )
        self.tfscript += self.api_gateway

    def build_lambda_role(self):
        """Builds AWS IAM Role with associated policies for the main lambda
        function."""

        self.lambda_iam_role = resource.aws_iam_role(
            f"pygrid-{self.config.app.name}-lambda-role",
            name=f"pygrid-{self.config.app.name}-lambda-role",
            assume_role_policy="""{
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": "sts:AssumeRole",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Effect": "Allow",
                        "Sid": ""
                    }
                ]
            }""",
            tags={"Name": f"pygrid-{self.config.app.name}-lambda-iam-role"},
        )
        self.tfscript += self.lambda_iam_role

        policy1 = resource.aws_iam_role_policy(
            "AWSLambdaVPCAccessExecutionRole",
            name="AWSLambdaVPCAccessExecutionRole",
            role=var(self.lambda_iam_role.id),
            policy=aws_lambda_vpc_execution_role_policy,
        )
        self.tfscript += policy1

        policy2 = resource.aws_iam_role_policy(
            "CloudWatchLogsFullAccess",
            name="CloudWatchLogsFullAccess",
            role=var(self.lambda_iam_role.id),
            policy=cloud_watch_logs_full_access_policy,
        )
        self.tfscript += policy2

        policy3 = resource.aws_iam_role_policy(
            "AmazonRDSDataFullAcess",
            name="AmazonRDSDataFullAcess",
            role=var(self.lambda_iam_role.id),
            policy=amazon_rds_data_full_access_policy,
        )
        self.tfscript += policy3

    def build_database(self):
        """Builds an Aurora serverless database."""

        db_parameter_group = resource.aws_db_parameter_group(
            "aurora_db_parameter_group",
            name=f"pygrid-{self.config.app.name}-aurora-db-parameter-group",
            family="aurora5.6",
            description=f"pygrid-{self.config.app.name}-aurora-db-parameter-group",
        )
        self.tfscript += db_parameter_group

        rds_cluster_parameter_group = resource.aws_rds_cluster_parameter_group(
            "aurora_cluster_56_parameter_group",
            name=f"pygrid-{self.config.app.name}-aurora-cluster-parameter-group",
            family="aurora5.6",
            description=f"pygrid-{self.config.app.name}-aurora-cluster-parameter-group",
        )
        self.tfscript += rds_cluster_parameter_group

        self.database = Module(
            "aurora",
            source="terraform-aws-modules/rds-aurora/aws",
            name=f"pygrid-{self.config.app.name}-database",
            engine="aurora",
            engine_mode="serverless",
            replica_scale_enabled=False,
            replica_count=0,
            subnets=[var(private_subnet.id) for private_subnet, _ in self.subnets],
            vpc_id=var(self.vpc.id),
            instance_type="db.t2.micro",
            enable_http_endpoint=True,  # Enable Data API,
            apply_immediately=True,
            skip_final_snapshot=True,
            storage_encrypted=True,
            database_name="pygridDB",
            username=self.config.credentials.db.username,
            password=self.config.credentials.db.password,
            db_parameter_group_name=var(db_parameter_group.id),
            db_cluster_parameter_group_name=var(rds_cluster_parameter_group.id),
            scaling_configuration={
                "auto_pause": True,
                "max_capacity": 64,  # ACU
                "min_capacity": 2,  # ACU
                "seconds_until_auto_pause": 300,
                "timeout_action": "ForceApplyCapacityChange",
            },
            tags={"Name": f"pygrid-{self.config.app.name}-aurora-database"},
        )
        self.tfscript += self.database

    def build_secret_manager(self):
        """Builds a secret manager which holds the database credentials."""

        random_pet = resource.random_pet("random", length=2)
        self.tfscript += random_pet

        self.db_secret_manager = resource.aws_secretsmanager_secret(
            "db-secret",
            name=f"pygrid-{self.config.app.name}-rds-{var(random_pet.id)}",
            description=f"PyGrid {self.config.app.name} database credentials",
            tags={"Name": f"pygrid-{self.config.app.name}-rds-secret-manager"},
        )
        self.tfscript += self.db_secret_manager

        db_secret_version = resource.aws_secretsmanager_secret_version(
            "db-secret-version",
            secret_id=var(self.db_secret_manager.id),
            secret_string=var(
                'jsonencode({"username" = "'
                + self.config.credentials.db.username
                + '", "password" = "'
                + self.config.credentials.db.password
                + '"})'
            ),
        )
        self.tfscript += db_secret_version

    def build_security_group(self):
        """Builds a security group for the lambda function."""

        self.security_group = resource.aws_security_group(
            "security_group",
            name="lambda-sg",
            vpc_id=var(self.vpc.id),
            ingress=[
                {
                    "description": "HTTPS",
                    "from_port": 443,
                    "to_port": 443,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
                {
                    "description": "HTTP",
                    "from_port": 80,
                    "to_port": 80,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
                {
                    "description": "PyGrid Nodes",
                    "from_port": 5000,
                    "to_port": 5999,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
                {
                    "description": "PyGrid Workers",
                    "from_port": 6000,
                    "to_port": 6999,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
                {
                    "description": "PyGrid Networks",
                    "from_port": 7000,
                    "to_port": 7999,
                    "protocol": "tcp",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                },
            ],
            egress=[
                {
                    "description": "Egress Connection",
                    "from_port": 0,
                    "to_port": 0,
                    "protocol": "-1",
                    "cidr_blocks": ["0.0.0.0/0"],
                    "ipv6_cidr_blocks": ["::/0"],
                    "prefix_list_ids": [],
                    "security_groups": [],
                    "self": False,
                }
            ],
            tags={"Name": "lambda-security-group"},
        )
        self.tfscript += self.security_group

    def build_lambda_function(self):
        """Builds the main lambda function hosting the app, and associate the
        lambda function with all the other deployed resources."""
        lambda_func = Module(
            "lambda",
            source="terraform-aws-modules/lambda/aws",
            function_name=f"pygrid-{self.config.app.name}",
            publish=True,  # To automate increasing versions
            runtime=self.python_runtime,
            source_path=f"{self.root_dir}/PyGrid/apps/{self.config.app.name}/src",
            handler="deploy.app",
            vpc_subnet_ids=[
                var(private_subnet.id) for private_subnet, _ in self.subnets
            ],
            vpc_security_group_ids=[var(self.security_group.id)],
            create_role=False,
            lambda_role=var(self.lambda_iam_role.arn),
            layers=[var(self.lambda_layer.arn)],
            environment_variables={
                "DB_NAME": self.database.database_name,
                "DB_CLUSTER_ARN": var_module(self.database, "this_rds_cluster_arn"),
                "DB_SECRET_ARN": var(self.db_secret_manager.arn),
            },
            allowed_triggers={
                "AllowExecutionFromAPIGateway": {
                    "service": "apigateway",
                    "source_arn": "{}/*/*".format(
                        var_module(
                            self.api_gateway, "this_apigatewayv2_api_execution_arn"
                        )
                    ),
                }
            },
            tags={"Name": f"pygrid-{self.config.app.name}-lambda-function"},
        )
        self.tfscript += lambda_func

        lambda_alias = Module(
            "lambda_alias",
            source="terraform-aws-modules/lambda/aws//modules/alias",
            name="prod",
            function_name=var_module(lambda_func, "this_lambda_function_name"),
            function_version=var_module(lambda_func, "this_lambda_function_version"),
        )
        self.tfscript += lambda_alias

    def zip_dependencies(self):
        """Clones the PyGrid repo and creates a zip file with the required
        dependencies."""
        pygrid_dir = os.path.join(self.root_dir, "PyGrid")

        bash_script = f"""
        if [ ! -d "{pygrid_dir}" ]
        then
            # Clone the required pygrid version
            mkdir {pygrid_dir}
            git clone https://github.com/OpenMined/PyGrid/ {pygrid_dir}
        fi

        if [ ! -f "{pygrid_dir}/{self.config.app.name}.zip" ]
        then
            # Let us first go to `apps/network` and export the poetry lock file to a requirements file.
            cd {pygrid_dir}/apps/{self.config.app.name}
            poetry export --format requirements.txt -o {pygrid_dir}/{self.config.app.name}_requirements.txt --without-hashes
            cd {pygrid_dir}

            # Build a zip file containing all dependencies of PyGrid Network, to deploy to an AWS Lambda Layer.
            # The root file should be called `Python`, and contains all the dependencies.
            mkdir python
            {self.python_runtime} -m pip install -r {pygrid_dir}/{self.config.app.name}_requirements.txt -t python
            zip -r {self.config.app.name}.zip python

            # Remove the temporary files and folders.
            rm -rf python {self.config.app.name}_requirements.txt
        fi
        """

        subprocess.call(bash_script, shell=True)
        return os.path.join(pygrid_dir, f"{self.config.app.name}.zip")
