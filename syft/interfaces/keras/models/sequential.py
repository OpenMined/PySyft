import syft.nn as nn
import sys
from syft.interfaces.keras.layers import Log
from syft import FloatTensor
import numpy as np
import json

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

		return self.syft.forward(input=x).to_numpy()

	def get_weights(self):
		return self.syft.parameters()

	def to_json(self):
		json_str = self.syft.to_json()
		# Postprocessing to match keras

		o = json.loads(json_str)

		o['config'][0]['config']['batch_input_shape'] = [None] + list(self.layers[0].input_shape)

		new_config = []
		for layer in o['config']:
			if layer["class_name"] == 'Linear':
				layer["class_name"] = 'Dense'
				layer["config"]["name"] = "dense_" + layer["config"]["name"].split("_")[-1]
			elif layer["class_name"] == "Softmax":
				new_config[-1]["config"]["activation"] = "softmax"
				continue
			elif layer["class_name"] == "ReLU":
				new_config[-1]["config"]["activation"] = "relu"
				continue

			new_config.append(layer)
		o['config'] = new_config

		return json.dumps(o)
