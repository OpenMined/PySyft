from . import base_worker
from .. import channels
from ..lib import strings

from ..processes.passively_broadcast_membership import PassivelyBroadcastMembershipProcess
from ..processes.listen_for_openmined_nodes import ListenForOpenMinedNodesProcess

class GridAnchor(base_worker.GridWorker):

	def __init__(self):
		super().__init__()

		self.node_type = "ANCHOR"

		# prints a picture of an anchor :)
		print(strings.anchor)

		# Blocking until this node has found at least one other OpenMined node
		# This functionality queries https://github.com/OpenMined/BootstrapNodes for Anchor nodes
		# then asks those nodes for which other OpenMined nodes they know about on the network.
		self.processes['listen_for_openmined_nodes'] = ListenForOpenMinedNodesProcess(self,min_om_nodes=0)

		# just lets the network know its a member of the openmined network
		self.processes['passively_broadcast_membership'] = PassivelyBroadcastMembershipProcess(self)

		
		