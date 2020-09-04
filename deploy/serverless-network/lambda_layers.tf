module "lambda_layer_pygrid_network_dependencies" {
  source = "terraform-aws-modules/lambda/aws"

  create_layer = true

  layer_name          = "lambda-layer-pygrid-network-dependencies"
  description         = "Lambda layer with all dependencies of PyGrid Network"
  compatible_runtimes = ["python3.6"]

  create_package = false
  s3_existing_package = {
    bucket = "bucket-with-pygrid-network-dependencies"
    key    = aws_s3_bucket_object.lambda_dependencies.id
  }
}
