from syft.workers.base import BaseWorker
from syft.federated.federated_client import FederatedClient


class VirtualWorker(BaseWorker, FederatedClient):
    def _send_msg(self, message: bin, location: BaseWorker, pending_time: float = 0) -> bin:
        if pending_time > 0:
            super().add_pending_time(pending_time)

        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        return self.recv_msg(message)
