from ..worker import Worker
from ..virtual.virtual_client import VirtualClient


class VirtualWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = VirtualClient(self.id, self, verbose=False)

    def _recv_msg(self, msg):
        return self.recv_msg(msg)

    def get_client(self, verbose=False):
        self.client.verbose = verbose
        return self.client
