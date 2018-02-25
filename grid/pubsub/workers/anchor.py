from . import base_worker
from .. import channels
from ...lib import strings

class GridAnchor(base_worker.GridWorker):

	def __init__(self):
		super().__init__()

		def just_listen():
			""

		print(strings.anchor)

		self.listen_to_channel(channels.openmined,just_listen)
		self.listen_to_channel(channels.list_workers,self.list_workers)
		self.listen_for_openmined_nodes(1)