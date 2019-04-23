from .base import BaseWorker


class VirtualWorker(BaseWorker):
    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        return self.recv_msg(message)
