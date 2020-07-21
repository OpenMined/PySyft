import enum
from .client import Client
from ...io.route import Route
from ...message.syft_message import SyftMessage

class NodeTypes(enum):
    Network = 1
    Domain = 2
    Device = 3
    VM = 4

class MyRemoteNodes(object):
    nodes = {}

    def __init__(self, my_type):
        for type in NodeTypes:
            self.nodes.update({type.name: []})
        self.iam = my_type

    def register_node(self, route: Route, type: str) -> None:
        if type not in self.nodes:
            # log unknown type
            return

        self.nodes[type].append(route)

    def forget_node(self, route: Route, type: str) -> None:
        self.nodes[type].pop(route)

    def get_node(self, type: str, id_or_name:(str, UID)):
        try:
            return self.nodes[type][id_or_name]
        except KeyError as e:
            try:
                id = self.vm_name2id[id_or_name]
                return self.vms[id]
            except KeyError as e:
                raise KeyError("You must ask for a vm using either a name or ID.")


    def route_message_to_relevant_nodes(self, message: SyftMessage) -> None:
        """
        check if the message should be forwarded.
        Network: routes to domains
        Domain: routes to devices
        Device: routes to VMs
        VM: doesn't route
        """
        route = message.route
        if self.iam == 'Network':
            if route.domain != '*':
                route.domain.client().send(message)
            else:
                for domain in self.nodes['domain']:
                    domain.client().send(message)
        elif self.iam == 'Domain':
            if route.device != '*':
                route.device.client().send(message)
            else:
                for device in self.nodes['device']:
                    device.client().send(message)
        elif self.iam == 'Device':
            if route.vm != '*':
                route.vm.client().send(message)
            else:
                for vm in self.nodes['vm']:
                    vm.client().send(message)

    def broadcast(self, route: Route):
        channel = route.broadcast_channel
        route.client().send_msg
