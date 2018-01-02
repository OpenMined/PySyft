import syft
import syft.nn as nn

class Sequential(object):

	def __init__(self):
		self.syft = nn.Sequential()

	def add(self, layer):

		# sometimes keras has single layers that actually correspond
		# to multiple syft layers - so they end up getting stored in 
		# an ordered list called "ordered_syft"
		for l in layer.ordered_syft:
			self.syft.add(l)