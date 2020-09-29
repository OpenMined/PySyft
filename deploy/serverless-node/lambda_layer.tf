locals {
  dependencies_zip_path    = "lambda-layer/pygrid-node-dep.zip"
}

resource "aws_s3_bucket" "node-lambda-layer-bucket" {
  bucket = "pygrid-node-lambda-layer-s3-bucket"
  acl    = "private"
  versioning {
    enabled = true
  }
}

resource "aws_s3_bucket_object" "node-lambda-layer" {
  bucket = aws_s3_bucket.node-lambda-layer-bucket.bucket
  key    = "${filemd5(local.dependencies_zip_path)}.zip"
  source = local.dependencies_zip_path
}

module "lambda_layer" {
  source = "terraform-aws-modules/lambda/aws"

  create_layer = true

  layer_name          = "pygrid-node-lambda-layer"
  compatible_runtimes = ["python3.8"]

  create_package = false
  s3_existing_package = {
    bucket = aws_s3_bucket_object.node-lambda-layer.bucket
    key    = aws_s3_bucket_object.node-lambda-layer.key
  }
}
