# Configure the AWS Provider
provider "aws" {
  version                 = "~> 3.0"
  region                  = var.aws_region
  shared_credentials_file = "$HOME/.aws/credentials"
}


# Create Virtual Private Cloud (VPC)
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr_block
  instance_tenancy     = "default"
  enable_dns_hostnames = true

  tags = {
    Name = "main-vpc"
  }
}

# Create Internet Gateway
resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "main-gw"
  }
}

# Create Route Table
resource "aws_route_table" "route_table" {
  vpc_id = aws_vpc.main.id

  route { # Default Route to GW
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }

  route {
    ipv6_cidr_block = "::/0"
    gateway_id      = aws_internet_gateway.gw.id
  }

  tags = {
    Name = "main-route-table"
  }
}

# Create subnet for webservers
resource "aws_subnet" "main" {
  vpc_id     = aws_vpc.main.id
  cidr_block = var.subnet_cidr_block

  tags = {
    Name = "main-subnet"
  }
}

# Associate subnet with the route table
resource "aws_route_table_association" "rta" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.route_table.id
}

# Create security group
resource "aws_security_group" "web" {
  name        = "allow_web_traffic"
  description = "Allow web inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "PyGrid Nodes"
    from_port   = 5000
    to_port     = 5999
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "PyGrid Workers"
    from_port   = 6000
    to_port     = 6999
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "PyGrid Networks"
    from_port   = 7000
    to_port     = 7999
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
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
    Name = "allow_web"
  }
}

resource "aws_instance" "webserver_instance" {
  for_each      = var.vm_names
  # TODO: future proof ami attribute
  # AWS can change ami names, so we need to make our ami attribute dynamic
  ami           = var.amis[var.aws_region]
  instance_type = var.instance_type
  key_name      = var.key_name

  associate_public_ip_address = true
  vpc_security_group_ids      = [aws_security_group.web.id]
  subnet_id                   = aws_subnet.main.id

  connection {
    type        = "ssh"
    user        = "ubuntu"
    host        = self.public_ip
    private_key = file(var.private_key)
    timeout     = "2m"
  }

  provisioner "remote-exec" {
    inline = [
      "echo ${each.key},${each.value} >> pygrid.txt"
    ]
  }

  user_data = file("deploy.sh")

  tags = {
    Name = each.key
  }
}


output "instance_id_list" {
  value = zipmap(values(aws_instance.webserver_instance)[*].tags.Name,
  values(aws_instance.webserver_instance)[*].public_ip)
}
