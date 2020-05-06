from time import sleep

import syft
from syft.workers.base import BaseWorker
from syft.federated.federated_client import FederatedClient


class VirtualWorker(BaseWorker, FederatedClient):
    def __init__(self, *args, **kwargs):
        super(VirtualWorker, self).__init__(*args, **kwargs)
        if kwargs["id"] != "me":
            hook = kwargs.get("hook")
            if isinstance(hook, syft.TorchHook):
                hook._virtual_workers.append(self)
            elif hook is None:
                for arg in args:
                    if isinstance(arg, syft.TorchHook):
                        arg._virtual_workers.append(self)

    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        """send message to worker location"""
        if self.message_pending_time > 0:
            if self.verbose:
                print(f"pending time of {self.message_pending_time} seconds to send message...")
            sleep(self.message_pending_time)

        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        """receive message"""
        return self.recv_msg(message)
