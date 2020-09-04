# Data sources to get VPC and subnets
data "aws_vpc" "default" {
  default = true
}

data "aws_subnet_ids" "all" {
  vpc_id = data.aws_vpc.default.id
}


module "aurora" {
  source = "terraform-aws-modules/rds-aurora/aws"

  name                  = "pygrid-network-database"
  engine                = "aurora"
  engine_mode           = "serverless"
  replica_scale_enabled = false
  replica_count         = 0

  subnets       = data.aws_subnet_ids.all.ids
  vpc_id        = data.aws_vpc.default.id
  instance_type = "db.r4.large"

  enable_http_endpoint = true   # Enable Data API

  apply_immediately               = true
  skip_final_snapshot             = true
  storage_encrypted               = true

  database_name = var.database_name
  username      = var.database_username
  password      = var.database_password

  db_parameter_group_name         = aws_db_parameter_group.aurora_db_56_parameter_group.id
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.aurora_cluster_56_parameter_group.id

  scaling_configuration = {
    auto_pause               = true
    max_capacity             = 64  #ACU
    min_capacity             = 2   #ACU
    seconds_until_auto_pause = 300
    timeout_action           = "ForceApplyCapacityChange"
  }
}

# Todo: Look into these.
resource "aws_db_parameter_group" "aurora_db_56_parameter_group" {
  name        = "test-aurora-db-56-parameter-group"
  family      = "aurora5.6"
  description = "test-aurora-db-56-parameter-group"
}

resource "aws_rds_cluster_parameter_group" "aurora_cluster_56_parameter_group" {
  name        = "test-aurora-56-cluster-parameter-group"
  family      = "aurora5.6"
  description = "test-aurora-56-cluster-parameter-group"
}