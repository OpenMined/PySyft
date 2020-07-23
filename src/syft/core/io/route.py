from ...decorators import syft_decorator
from ...common.id import UID
from typing import final


class BaseRoute(object):
    connection_details = {}

    def configure_connection(self, protocol: str, host: str, port: int):
        """
        the route should have connection details embedded in it.
        so that nodes operators can utilize it to route messages.
        """
        self.connection_details.update({
            'host': host,
            'protocol': protocol,
            'port': port
        })

    def register_broadcast_channel(self, channel_name: str):
        """
        In the case configured protocol is pub/sub or event driven.
        Args:
            channel_name: the name of the channel to broadcast on.
        """
        self.connection_details.update({'broadcast_channel': name})

    def generate_token(self):
        """
        could potentially provide a token that declares the
        permissions on this route.
        """
        pass


@final
class PublicRoute(BaseRoute):
    @syft_decorator(typechecking=True)
    def __init__(self, network: (str, UID), domain: (str, UID)):
        self.network = network
        self.domain = domain


@final
class PrivateRoute(BaseRoute):
    @syft_decorator(typechecking=True)
    def __init__(self, device: (str, UID), vm: (str, UID)):
        self.device = device
        self.vm = vm


@final
class Route(BaseRoute):
    @syft_decorator(typechecking=True)
    def __init__(self, pub_route: PublicRoute, pri_route: PrivateRoute):
        self.pub_route = pub_route
        self.pri_route = pri_route


@syft_decorator(typechecking=True)
def route(network: (str, UID), domain: (str, UID), device: (str, UID), vm: (str, UID)) -> Route:
    """A convenience method for creating routes"""

    pub = PublicRoute(network=network, domain=domain)
    pri = PrivateRoute(device=device, vm=vm)

    return Route(pub_route=pub, pri_route=pri)
