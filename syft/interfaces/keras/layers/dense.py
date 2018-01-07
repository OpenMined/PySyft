import syft
import syft.nn as nn

import numpy as np

class Dense(object):

	def __init__(self, units, input_shape=None, activation=None):
		self.units = units
		self.input_shape = input_shape
		self.output_shape = self.units
		self.activation_str = activation

		# sometimes keras has single layers that actually correspond
		# to multiple syft layers - so they end up getting stored in 
		# an ordered list called "ordered_syft"
		self.ordered_syft = list()

		if(input_shape != None):
			self.create_model()

	def create_model(self):

		self.syft_model = nn.Linear(int(np.sum(self.input_shape)),self.units)
		self.ordered_syft.append(self.syft_model)

		if(self.activation_str != None):
			if(self.activation_str == 'relu'):
				self.syft_activation = nn.ReLU()
			elif(self.activation_str == 'softmax'):
				self.syft_activation = nn.Softmax()
			self.ordered_syft.append(self.syft_activation)
		else:
			self.syft_activation = None

		
