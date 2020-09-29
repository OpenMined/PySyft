provider "aws" {
  region                  = "us-east-1"
  shared_credentials_file = "$HOME/.aws/credentials"
}

# Create Virtual Private Cloud (VPC)
resource "aws_vpc" "pygrid_node" {
  cidr_block           = "10.0.0.0/26"
  instance_tenancy     = "default"
  enable_dns_hostnames = true

  tags = {
    Name = "pygrid-node-vpc"
  }
}

# --Create private subnets for lambda-- #

resource "aws_subnet" "private_subnet_1" {
  vpc_id            = aws_vpc.pygrid_node.id
  cidr_block        = "10.0.0.0/28"
  availability_zone = "us-east-1a"
  tags = {
    Name = "GridNode-private-1"
  }
}

resource "aws_subnet" "private_subnet_2" {
  vpc_id            = aws_vpc.pygrid_node.id
  cidr_block        = "10.0.0.16/28"
  availability_zone = "us-east-1b"
  tags = {
    Name = "GridNode-private-2"
  }
}

# Create public subnet
resource "aws_subnet" "public_subnet" {
  vpc_id            = aws_vpc.pygrid_node.id
  cidr_block        = "10.0.0.32/28"
  availability_zone = "us-east-1c"
  tags = {
    Name = "GridNode-Public"
  }
}


