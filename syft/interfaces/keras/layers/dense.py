import syft
import syft.nn as nn

import numpy as np

class Dense(object):

	def __init__(self, units, input_shape, activation=None):
		self.units = units
		self.input_shape = input_shape
		self.activation_str = activation

		# sometimes keras has single layers that actually correspond
		# to multiple syft layers - so they end up getting stored in 
		# an ordered list called "ordered_syft"
		self.ordered_syft = list()


		self.syft_model = nn.Linear(int(np.sum(input_shape)),units)
		self.ordered_syft.append(self.syft_model)

		if(activation):
			self.syft_activation = nn.ReLU()
			self.ordered_syft.append(self.syft_activation)
		else:
			self.syft_activation = None

		
