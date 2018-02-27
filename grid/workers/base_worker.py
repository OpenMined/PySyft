from .. import base
from ..services.broadcast_known_workers import BroadcastKnownWorkersService

class GridWorker(base.PubSub):

    def __init__(self):
        super().__init__('worker')

        # LAUNCH SERVICES - these are non-blocking and run on their own threads

        # all service objects will live in this dictionary
        self.services = {}

        # this service serves the purpose of helping other nodes find out about nodes on the network.
        # if someone queries the "list_worker" channel - it'll send a message directly to the querying node
        # with a list of the OpenMined nodes of which it is aware.
        self.services['broadcast_known_workers'] = BroadcastKnownWorkersService(self)



