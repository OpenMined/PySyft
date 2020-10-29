from .aws import *


class AWS_Serverfull(AWS):
    def __init__(self, credentials, vpc_config, db_config, app_config) -> None:
        """
        credentials (dict) : Contains AWS credentials
        vpc_config (dict) : Contains arguments required to deploy the VPC
        db_config (dict) : Contains username and password of the deployed database
        app_config (dict) : Contains arguments which are required to deploy the app.
        """

        super().__init__(credentials, vpc_config)

        self.app = app_config["name"]

        self.db_username = db_config["username"]
        self.db_password = db_config["password"]

        self.deploy()

    def deploy(self):
        pass
