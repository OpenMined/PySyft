from time import sleep

from syft.workers.base import BaseWorker
from syft.federated.federated_client import FederatedClient


class VirtualWorker(BaseWorker, FederatedClient):
    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        """send message to worker location"""
        # if self.message_pending_time > 0:
        #     #if self.verbose:
        #     print(f"pending time of {self.message_pending_time} seconds to send message...")
        #     sleep(self.message_pending_time)

        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        if self.id == "alice":
            if not hasattr(self, "count"):
                self.count = False
            if self.count:
                if not hasattr(self, "received_load"):
                    self.received_load = 0
                message_size = len(message)
                self.received_load += message_size
                print("tot:", self.received_load, "\tmsg:", message_size)

        if self.message_pending_time > 0:
            # if self.verbose:
            print(f"pending time of {self.message_pending_time} seconds to send message...")
            sleep(self.message_pending_time)
        """receive message"""
        return self.recv_msg(message)
