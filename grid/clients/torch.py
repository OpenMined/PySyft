import torch
from torch.autograd import Variable
import inspect
import random
import copy
from ..workers import client
from ..lib import utils
from ..lib import serde
from ..services.torch.torch_service import TorchService
import json

class TorchClient(client.BaseClient):

    def __init__(self,min_om_nodes=1,known_workers=list(),include_github_known_workers=True):
        super().__init__(min_om_nodes=min_om_nodes,
                        known_workers=known_workers,
                        include_github_known_workers=include_github_known_workers)


        self.services['torch_service'] = TorchService(self)

        
    