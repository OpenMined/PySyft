locals {
  function_path    = "../../apps/node/src/"
  function_handler = "deploy.app"
}

resource "aws_security_group" "lambda_sg" {
  vpc_id = aws_vpc.pygrid_node.id

  ingress {
    description = "Allow EFS"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "lambda-sg"
  }
}

module "lambda" {
  source = "terraform-aws-modules/lambda/aws"

  function_name = "pygrid-node"
  description   = "Node hosted by UCSF"
  publish       = true # To automate increasing versions

  runtime     = "python3.8"
  source_path = local.function_path
  handler     = local.function_handler

  timeout     = 60 * 3 # 3 minutes
  memory_size = 1500   # 1500 MB

  create_role = false
  lambda_role = aws_iam_role.pygrid-node-lambda-role.arn

  layers = [
    module.lambda_layer.this_lambda_layer_arn,
  ]

  vpc_subnet_ids         = [aws_subnet.private_subnet_1.id, aws_subnet.private_subnet_2.id]
  vpc_security_group_ids = [aws_security_group.lambda_sg.id]

  file_system_arn              = aws_efs_access_point.node-access-points.arn
  file_system_local_mount_path = "/mnt${var.mount_path}"


  allowed_triggers = {
    AllowExecutionFromAPIGateway = {
      service    = "apigateway"
      source_arn = "${module.api_gateway.this_apigatewayv2_api_execution_arn}/*/*"
    }
  }

  environment_variables = {
    MOUNT_PATH     = "/mnt${var.mount_path}"
    DB_NAME        = var.database_name
    DB_CLUSTER_ARN = module.aurora.this_rds_cluster_arn
    DB_SECRET_ARN  = aws_secretsmanager_secret.database-secret.arn
  }

}

module "lambda_alias" {
  source = "terraform-aws-modules/lambda/aws//modules/alias"

  name          = "prod"
  function_name = module.lambda.this_lambda_function_name

  # Set function_version when creating alias to be able to deploy using it,
  # because AWS CodeDeploy doesn't understand $LATEST as CurrentVersion.
  function_version = module.lambda.this_lambda_function_version
}
