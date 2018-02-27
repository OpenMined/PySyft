from . import base_worker
from ...lib import strings, utils, output_pipe
from .. import channels
from ..processes.fit_worker import FitWorkerProcess
import json
import threading

class GridCompute(base_worker.GridWorker):

    """
    This class runs a worker whose purpose is to do the following:
       - PRIMARY: use local compute resources to train models at the request of clients on the network
       - SECONDARY: learn about the existence of other nodes on the network - and help others to do so when asked
    """


    def __init__(self):
        super().__init__()

        self.node_type = "COMPUTE"

        # prints a pretty picture of a Computer
        print(strings.compute)

        # Blocking until this node has found at least one other OpenMined node
        # This functionality queries https://github.com/OpenMined/BootstrapNodes for Anchor nodes
        # then asks those nodes for which other OpenMined nodes they know about on the network.
        self.listen_for_openmined_nodes(1)

        # This process listens for models that it can train.
        self.fit_worker_process = FitWorkerProcess(self)        

    