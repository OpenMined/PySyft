from syft.workers.base import BaseWorker
from syft.federated import FederatedClient


class VirtualWorker(BaseWorker, FederatedClient):
    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        return self.recv_msg(message)
