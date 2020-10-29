import math


def generate_cidr_block(num_ip_addresses, num_subnets):
    cidr_block_step = int(num_ip_addresses / num_subnets)
    subnet_ip_prefix = 32 - int(math.log(cidr_block_step, 2))
    for i in range(num_subnets):
        yield f"10.0.0.{cidr_block_step * i}/{subnet_ip_prefix}"
