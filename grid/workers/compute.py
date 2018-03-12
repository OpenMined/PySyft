from . import base_worker
from ..lib import strings, utils, output_pipe
from .. import channels
from ..services.fit_worker import FitWorkerService
from ..services.listen_for_openmined_nodes import ListenForOpenMinedNodesService
from ..services.torch.torch_service import TorchService
import json
import threading


class GridCompute(base_worker.GridWorker):
    """
    This class runs a worker whose purpose is to do the following:
       - PRIMARY: use local compute resources to train models at the request of clients on the network
       - SECONDARY: learn about the existence of other nodes on the network - and help others to do so when asked
    """

    def __init__(self):
        super().__init__('COMPUTE')

        # prints a pretty picture of a Computer
        print(strings.compute)

        # Blocking until this node has found at least one other OpenMined node
        # This functionality queries https://github.com/OpenMined/BootstrapNodes for Anchor nodes
        # then asks those nodes for which other OpenMined nodes they know about on the network.
        self.services[
            'listen_for_openmined_nodes'] = ListenForOpenMinedNodesService(
                self, min_om_nodes=1)

        # KERAS

        # This process listens for models that it can train.
        self.services['fit_worker_service'] = FitWorkerService(self)

        # TORCH

        # this process listens for torch ops
        self.services['torch_service'] = TorchService(self)
