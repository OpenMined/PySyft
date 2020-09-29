# Internet Gateway
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.pygrid_node.id

  tags = {
    Name = "pygrid-node-igw"
  }
}

# EIP for NAT Gateway
resource "aws_eip" "eip" {
  vpc = true
}


# NAT Gateway
resource "aws_nat_gateway" "ngw" {
  allocation_id = aws_eip.eip.id
  subnet_id     = aws_subnet.public_subnet.id

  tags = {
    Name = "pygrid-node-ngw"
  }
}


# Route Table for Public Subnet
resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.pygrid_node.id

  route {
    # Route to Internet GW
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "pygrid-node-public-RT"
  }
}

# Route Table for Private Subnet
resource "aws_route_table" "private_route_table" {
  vpc_id = aws_vpc.pygrid_node.id

  route {
    # Route to NAT GW
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_nat_gateway.ngw.id
  }

  tags = {
    Name = "pygrid-node-private-RT"
  }
}

# Associate subnet with the route table
resource "aws_route_table_association" "rta_public_subnet" {
  subnet_id      = aws_subnet.public_subnet.id
  route_table_id = aws_route_table.public_route_table.id
}

resource "aws_route_table_association" "rta_private_subnet_1" {
  subnet_id      = aws_subnet.private_subnet_1.id
  route_table_id = aws_route_table.private_route_table.id
}

resource "aws_route_table_association" "rta_private_subnet_2" {
  subnet_id      = aws_subnet.private_subnet_2.id
  route_table_id = aws_route_table.private_route_table.id
}
