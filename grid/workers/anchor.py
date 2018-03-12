from . import base_worker
from ..lib import strings

from ..services.passively_broadcast_membership import PassivelyBroadcastMembershipService
from ..services.listen_for_openmined_nodes import ListenForOpenMinedNodesService


class GridAnchor(base_worker.GridWorker):
    def __init__(self):
        super().__init__('ANCHOR')

        # prints a picture of an anchor :)
        print(strings.anchor)

        # Blocking until this node has found at least one other OpenMined node
        # This functionality queries https://github.com/OpenMined/BootstrapNodes for Anchor nodes
        # then asks those nodes for which other OpenMined nodes they know about on the network.
        self.services[
            'listen_for_openmined_nodes'] = ListenForOpenMinedNodesService(
                self, min_om_nodes=0)

        # just lets the network know its a member of the openmined network
        self.services[
            'passively_broadcast_membership'] = PassivelyBroadcastMembershipService(
                self)
