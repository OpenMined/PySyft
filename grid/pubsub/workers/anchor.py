from . import base_worker
from .. import channels
from ...lib import strings

class GridAnchor(base_worker.GridWorker):

	def __init__(self):
		super().__init__()

		def just_listen():
			""

		print(strings.anchor)

		# Blocking until this node has found at least one other OpenMined node
		# This functionality queries https://github.com/OpenMined/BootstrapNodes for Anchor nodes
		# then asks those nodes for which other OpenMined nodes they know about on the network.
		self.listen_for_openmined_nodes(min_om_nodes=int(0))

		self.listen_to_channel(channels.openmined,just_listen)
		self.listen_to_channel(channels.list_workers,self.list_workers)
		