from ..abstract.client import Client


class DomainClient(Client):
    def __init__(self, id, connection):
        super().__init__(id=id, connection=connection)
        self.connection

    def __repr__(self):
        return f"<DomainClient id:{self.id}>"

    def get_connection_route(self, network, domain = None, device = None, vm = None):
        """
        e.g. when a network requests to connect to a domain,
        it should use this method.

        here we should have checks and verifications
        """
        route = Route(network=network, domain=domain, device=device, vm=vm)
        route.configure_connection(self)
        return route
