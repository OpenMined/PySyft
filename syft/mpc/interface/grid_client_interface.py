from .base_interface import BaseInterface


class GridClientInterface(BaseInterface):

    def __init__(self, party, grid_client):
        super.__init__(self, party)
        raise NotImplementedError()

    def send(self, var):
        raise NotImplementedError()

    def recv(self, var):
        raise NotImplementedError()
