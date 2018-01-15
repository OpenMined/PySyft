import syft.nn as nn
import sys
from syft.interfaces.keras.layers import Log
from syft import FloatTensor
import numpy as np

class Sequential(object):

	def __init__(self):
		self.syft = nn.Sequential()
		self.layers = list()
		self.compiled = False

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

	def compile(self,loss,optimizer,metrics):
		if(not self.compiled):
			self.compiled = True

			if(loss == 'categorical_crossentropy'):
				self.loss = nn.CategoricalCrossEntropy()
			elif(loss == 'mean_squared_error'):
				self.loss = nn.MSELoss()

			self.optimizer = optimizer
			self.metrics = metrics

			self.optimizer.init(syft_params=self.syft.parameters())
		else:
			sys.stderr.write("Warning: Model already compiled... please rebuild from scratch if you need to change things")


	def fit(self, x_train, y_train, batch_size, epochs=1, validation_data=None, log_interval=1, verbose=False):
		return self.syft.fit(input=x_train, target=y_train, batch_size=batch_size, criterion=self.loss,
							 optim=self.optimizer.syft, iters=epochs, log_interval=log_interval, verbose=verbose)

	def evaluate(self, test_input, test_target, batch_size, metrics=[], verbose=True):
		return self.syft.evaluate(test_input, test_target, self.loss, metrics=metrics, verbose=verbose, batch_size=batch_size)

	def predict(self,x):

		if(type(x) == list):
			x = np.array(x).astype('float')
		if(type(x) == np.array or type(x) == np.ndarray):
			x = FloatTensor(x,autograd=True, delete_after_use=False)

		return self.syft.forward(input=x)

	def get_weights(self):
		return self.syft.parameters()
