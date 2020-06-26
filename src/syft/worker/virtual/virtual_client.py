from syft.worker.client import Client


class VirtualClient(Client):
    def __init__(self, id, worker, verbose=False):
        super().__init__(id=id, verbose=verbose)
        self.worker = worker

    def _send_msg(self, msg):
        return self.worker._recv_msg(msg)

    def send_msg(self, msg):
        return self._send_msg(msg)

    def __repr__(self):
        out = f"<VirtualClient id:{self.id}>"

        if self.verbose:
            out += "\n" + str(self.worker.store._objects)
        return out
