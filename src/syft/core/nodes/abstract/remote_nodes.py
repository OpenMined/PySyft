import enum
from .client import Client
from ...io.route import Route
from ...message.syft_message import SyftMessage


class MyRemoteNodes(object):
    nodes = {}

    def __init__(self, my_route: Route):
        for type in RouteTypes:
            self.nodes.update({type.name: []})
        self.iam = iam.type
        self.my_route = my_route

    def register_node(self, route: Route, type: RouteTypes) -> None:
        if type not in self.nodes:
            # log unknown type
            return

        self.nodes[type].append(route)

    def forget_node(self, route: Route, type: RouteTypes) -> None:
        self.nodes[type].pop(route)

    def route_message_to_relevant_nodes(self, route: Route, message: SyftMessage) -> None:
        """
        check if the message should be forwarded.
        Network: routes to domains
        Domain: routes to devices
        Device: routes to VMs
        VM: doesn't route
        """
        if self.iam == 'network':
            if route.domain != '*':
                route.domain.client().send(message)
            else:
                for domain in self.nodes['domain']:
                    domain.client().send(message)
        elif self.iam == 'domain':
            if route.device != '*':
                route.device.client().send(message)
            else:
                for device in self.nodes['device']:
                    device.client().send(message)
        elif self.iam == 'device':
            if route.vm != '*':
                route.vm.client().send(message)
            else:
                for vm in self.nodes['vm']:
                    vm.client().send(message)

    def broadcast(self, route: Route):
        channel = route.broadcast_channel
