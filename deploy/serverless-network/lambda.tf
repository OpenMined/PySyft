locals {
  # TODO: Avoid __pycache__ folders to be zipped
  function_path    = "../../apps/network/src/"
  function_handler = "deploy.app"
}

module "lambda" {
  source = "terraform-aws-modules/lambda/aws"

  function_name = "pygrid-network"
  description   = "Network hosted by American Medical Association"
  publish       = true    # To automate increasing versions

  runtime     = "python3.6"
  source_path = local.function_path
  handler     = local.function_handler

  create_role = false
  lambda_role = aws_iam_role.pygrid-network-lambda-role.arn

  layers = [
    module.lambda_layer_pygrid_network_dependencies.this_lambda_layer_arn,
  ]

  environment_variables = {
    DB_NAME = var.database_name
    DB_CLUSTER_ARN = module.aurora.this_rds_cluster_arn
    DB_SECRET_ARN  = aws_secretsmanager_secret.database-secret.arn
    SECRET_KEY = "Do-we-need-this-in-deployed-version"  # TODO: Clarify this
  }

  allowed_triggers = {
    AllowExecutionFromAPIGateway = {
      service    = "apigateway"
      source_arn = "${module.api_gateway.this_apigatewayv2_api_execution_arn}/*/*"
    }
  }
}

module "lambda_alias" {
  source = "terraform-aws-modules/lambda/aws//modules/alias"

  name = "prod"

  function_name = module.lambda.this_lambda_function_name

  # Set function_version when creating alias to be able to deploy using it,
  # because AWS CodeDeploy doesn't understand $LATEST as CurrentVersion.
  function_version = module.lambda.this_lambda_function_version
}