from .base_interface import BaseInterface


class GridWorkerInterface(BaseInterface):
    def __init__(self, party, grid_worker):
        super.__init__(self, party)
        raise NotImplementedError()

    def send(self, var):
        raise NotImplementedError()

    def recv(self, var):
        raise NotImplementedError()
