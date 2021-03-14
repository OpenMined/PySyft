aws_lambda_vpc_execution_role_policy = """{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "ec2:CreateNetworkInterface",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DeleteNetworkInterface"
                    ],
                    "Resource": "*"
                }
            ]
        }"""

cloud_watch_logs_full_access_policy = """{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": [
                        "logs:*"
                    ],
                    "Effect": "Allow",
                    "Resource": "*"
                }
            ]
        }
        """

amazon_rds_data_full_access_policy = """{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "SecretsManagerDbCredentialsAccess",
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:PutResourcePolicy",
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:DeleteSecret",
                    "secretsmanager:DescribeSecret",
                    "secretsmanager:TagResource"
                ],
                "Resource": "*"
            },
            {
                "Sid": "RDSDataServiceAccess",
                "Effect": "Allow",
                "Action": [
                    "dbqms:CreateFavoriteQuery",
                    "dbqms:DescribeFavoriteQueries",
                    "dbqms:UpdateFavoriteQuery",
                    "dbqms:DeleteFavoriteQueries",
                    "dbqms:GetQueryString",
                    "dbqms:CreateQueryHistory",
                    "dbqms:DescribeQueryHistory",
                    "dbqms:UpdateQueryHistory",
                    "dbqms:DeleteQueryHistory",
                    "rds-data:ExecuteSql",
                    "rds-data:ExecuteStatement",
                    "rds-data:BatchExecuteStatement",
                    "rds-data:BeginTransaction",
                    "rds-data:CommitTransaction",
                    "rds-data:RollbackTransaction",
                    "secretsmanager:CreateSecret",
                    "secretsmanager:ListSecrets",
                    "secretsmanager:GetRandomPassword",
                    "tag:GetResources"
                ],
                "Resource": "*"
            }
        ]
     }
     """
