from ...decorators import syft_decorator
from ...common.id import UID
from typing import final


class BaseRoute(object):
    def configure_connection(self, connection_details):
        """
        the route should have connection details embedded in it.
        """
        self.connection_details = connection_details

    def register_broadcast_channel(self, channel_name):
        """
        In the case configured protocol is pub/sub or event driven.
        Args:
            channel_name: the name of the channel to broadcast on.
        """
        self.broadcast_channel = name

    def client(self):
        # connect to configured connection and return a client obj.
        return self.client


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
