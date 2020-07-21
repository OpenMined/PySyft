from ..abstract.client import Client


class DomainClient(Client):
    def __init__(self, id, connection):
        super().__init__(id=id, connection=connection)
        self.connection

    def __repr__(self):
        return f"<DomainClient id:{self.id}>"
