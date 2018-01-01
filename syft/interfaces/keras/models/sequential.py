import syft
import syft.nn as nn

class Sequential(object):

	def __init__(self):
		self.syft = nn.Sequential()
		self.layers = list()

	def add(self, layer):

		if(len(self.layers) > 0):

			# look to the previous layer to get the input shape for this layer
			layer.input_shape = self.layers[-1].output_shape

			# if layer doesn't know its output shape - it's probably dynamic
			if not hasattr(layer, 'output_shape'):
				layer.output_shape = layer.input_shape

			layer.create_model()

		self.layers.append(layer)

		# sometimes keras has single layers that actually correspond
		# to multiple syft layers - so they end up getting stored in 
		# an ordered list called "ordered_syft"
		for l in layer.ordered_syft:
			self.syft.add(l)

	def summary(self):
		self.syft.summary()

	def compile(loss,optimizer,metrics):
		self.loss = loss
		self.optimizer = optimizer
		self.metrics = metrics