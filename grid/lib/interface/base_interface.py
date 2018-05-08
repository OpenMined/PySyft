from abc import ABC, abstractmethod


class BaseInterface(ABC):

    def __init__(self, party):
        self.party = party
        if party:
            self.other = 0
        else:
            self.other = 1

    @abstractmethod
    def send(self, var):
        pass

    @abstractmethod
    def recv(self,var):
        pass

    def get_party(self):
        return self.party
