# # stdlib
# stdlib
from typing import Any
from typing import List

# # relative
# from ..core.node.domain_client import DomainClient


class DatasetId():
    value: str
    name: str
    

class Client():
    login: Any #Will use this for DomainClient
    datasets: list = []
    
    def __init__(self,login: Any=None, datasets=[]):
        self.login=login,
        self.datasets=datasets


class DeploymentClient():
    deployment_id: str
    user_key_name: str
    connection_port: int = 3032
    
    def __init__(self,deployment_id: Any=None, user_key_name="" , connection_port=3032):
        self.deployment_id=deployment_id
        self.user_key_name = user_key_name
        self.connection_port = connection_port

# # class Expression():
    



