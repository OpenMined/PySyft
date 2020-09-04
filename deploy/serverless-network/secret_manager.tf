resource "aws_secretsmanager_secret" "database-secret" {
  name                    = "pygrid-network-rds-admin-${random_pet.random.id}"
  description             = "PyGrid network database credentials"
}


resource "aws_secretsmanager_secret_version" "secret" {
  secret_id     = aws_secretsmanager_secret.database-secret.id
  secret_string = jsonencode({"username"=var.database_username, "password"=var.database_password})
}
