resource "aws_efs_file_system" "pygrid-syft-dependenices" {
  creation_token = "pygrid-syft-dependencies"

  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  lifecycle_policy {
    transition_to_ia = "AFTER_90_DAYS"
  }

  tags = {
    Name = "pygrid-syft-dependencies"
  }
}

# Mount target connects the file system to the subnet
resource "aws_efs_mount_target" "efs-mt-1" {
  file_system_id  = aws_efs_file_system.pygrid-syft-dependenices.id
  security_groups = [aws_security_group.lambda_sg.id]
  subnet_id       = aws_subnet.private_subnet_1.id
}

resource "aws_efs_mount_target" "efs-mt-2" {
  file_system_id  = aws_efs_file_system.pygrid-syft-dependenices.id
  security_groups = [aws_security_group.lambda_sg.id]
  subnet_id       = aws_subnet.private_subnet_2.id
}

resource "aws_efs_mount_target" "efs-mt-3" {
  file_system_id  = aws_efs_file_system.pygrid-syft-dependenices.id
  security_groups = [aws_security_group.lambda_sg.id]
  subnet_id       = aws_subnet.public_subnet.id
}

resource "aws_efs_access_point" "node-access-points" {
  file_system_id = aws_efs_file_system.pygrid-syft-dependenices.id

  root_directory {
    path = var.mount_path
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "777"
    }
  }

  posix_user {
    gid = 1000
    uid = 1000
  }

  tags = {
    Name = "access_point_for_lambda"
  }

  depends_on = [
    aws_efs_file_system.pygrid-syft-dependenices,
    aws_efs_mount_target.efs-mt-1,
    aws_efs_mount_target.efs-mt-2
  ]
}


