from ..nodes.abstract.remote_nodes import RemoteNodes
from ..device.client import DomainClient

class NetworkRemoteNodes(RemoteNodes):
    def route_message_to_relevant_nodes(self, message):
        """
        if the message has any public routes that don't point to me.
        I should forward it to them.
        Otherwise I should forward to domains, if domains are specified.
        """
        pass
