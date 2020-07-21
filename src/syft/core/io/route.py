from ...decorators import syft_decorator
from ...common.id import UID
from typing import final


class BaseRoute(object):
    def configure_connection(self, protocol, url, port):
        pass

    def register_broadcast_channel(self, channel_name):
        """
        Args:
            channel_name: the name of the channel to broadcast on.
        """
        self.broadcast_channel = name

    def client(self):
        # connect to configured connection.
        return

    def type(self):
        if self.vm != '*':
            return 'VM'
        if self.domain != '*':
            return 'Domain'
        if self.device != '*':
            return 'Device'
        if self.network != '*':
            return 'Network'

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
