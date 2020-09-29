resource "aws_secretsmanager_secret" "database-secret" {
  name        = "pygrid-node-rds-admin-${random_pet.random.id}"
  description = "PyGrid node database credentials"
}


resource "aws_secretsmanager_secret_version" "secret" {
  secret_id     = aws_secretsmanager_secret.database-secret.id
  secret_string = jsonencode({ "username" = var.database_username, "password" = var.database_password })
}


resource "random_pet" "random" {
  length = 2
}