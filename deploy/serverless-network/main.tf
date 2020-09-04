provider "aws" {
  region                  = "us-east-1"
  shared_credentials_file = "$HOME/.aws/credentials"
}


module "api_gateway" {
  source = "terraform-aws-modules/apigateway-v2/aws"

  name          = "PygridNetwokrAPIGateway-http"
  description   = "PyGrid Network HTTP API Gateway"
  protocol_type = "HTTP"

  create_api_domain_name = false

  integrations = {
    "$default" = {
      lambda_arn = module.lambda.this_lambda_function_arn
    }
  }
}

resource "random_pet" "random" {
  length = 2
}