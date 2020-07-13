from ...typecheck import type_hints
from typing import final


@final
class PublicRoute(object):
    @type_hints
    def __init__(self, network: str, domain: str):
        self.network = network
        self.domain = domain


@final
class PrivateRoute(object):
    @type_hints
    def __init__(self, device: str, vm: str):
        self.device = device
        self.vm = vm


@final
class Route(object):
    @type_hints
    def __init__(self, pub_route: PublicRoute, pri_route: PrivateRoute):
        self.pub_route = pub_route
        self.pri_route = pri_route


@type_hints
def route(network: str, domain: str, device: str, vm: str) -> Route:
    """A convenience method for creating routes"""

    pub = PublicRoute(network=network, domain=domain)
    pri = PrivateRoute(device=device, vm=vm)

    return Route(pub_route=pub, pri_route=pri)
