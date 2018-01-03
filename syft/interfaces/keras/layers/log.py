import syft
import syft.nn as nn

import numpy as np

class Log(object):

	def __init__(self):
		
		# sometimes keras has single layers that actually correspond
		# to multiple syft layers - so they end up getting stored in 
		# an ordered list called "ordered_syft"
		self.ordered_syft = list()
		self.syft_model = nn.Log()
		self.ordered_syft.append(self.syft_model)

	def create_model(self):
		""
		

		
