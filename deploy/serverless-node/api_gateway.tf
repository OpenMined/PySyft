module "api_gateway" {
  source = "terraform-aws-modules/apigateway-v2/aws"

  name          = "PygridNodeAPIGateway-http"
  description   = "Node HTTP API Gateway"
  protocol_type = "HTTP"

  create_api_domain_name = false

  integrations = {
    "$default" = {
      lambda_arn = module.lambda.this_lambda_function_arn
    }
  }
}