from .base import BaseWorker


class VirtualWorker(BaseWorker):
    def _send_msg(self, message, location):
        return location._recv_msg(message)

    def _recv_msg(self, message):
        return self.recv_msg(message)
