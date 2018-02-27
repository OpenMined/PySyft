from .. import channels
from .base import BaseProcess

class PassivelyBroadcastMembershipProcess(BaseProcess):

	# this process just listens on the general "openmined" channel so that other nodes
	# on the network know its there.

	def __init__(self,worker):
		super().__init__(worker)

		def just_listen():
			""
		self.listen_to_channel(channels.openmined,just_listen)