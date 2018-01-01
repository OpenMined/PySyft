import syft.controller as controller
import sys

class Model():
	def __init__(self, id=None):
		self.sc = controller
		self.params = False
		self.type = None
		self._layer_type = None
		self.id = id
		self.type = "model"
		self.output_shape = "(dynamic)"

	def init(self,layer_type,params=[]):
		self.type = "model"
		self._layer_type = layer_type
		self.sc = controller
		self.id = -1
		self.id = int(self.sc.send_json(self.cmd("create",[self._layer_type] + params)))

	def discover(self):

		self._layer_type = self.layer_type()
		if(self._layer_type == 'linear'):
			return Linear(id = self.id)
		elif(self._layer_type == 'sigmoid'):
			return Sigmoid(id = self.id)
		elif(self._layer_type == 'crossentropyloss'):
			return CrossEntropyLoss(id = self.id)
		elif(self._layer_type == 'tanh'):
			return Tanh(id = self.id)
		elif(self._layer_type == 'dropout'):
			return Dropout(id = self.id)
		elif(self._layer_type == 'softmax'):
			return Softmax(id = self.id)
		elif(self._layer_type == 'logsoftmax'):
			return LogSoftmax(id = self.id)
		elif(self._layer_type == 'relu'):
			return ReLU(id = self.id)
		elif(self._layer_type == 'log'):
			return Log(id = self.id)
		else:
			sys.stderr.write("Attempted to discover the type - but it wasn't supported. Has the layer type '"
			 + self._layer_type + "' been added to the discover() method in nn.py?")

	def __call__(self,*args):
		if(len(args) == 1):
			return self.forward(args[0])
		elif(len(args) == 2):
			return self.forward(args[0],args[1])
		elif(len(args) == 3):
			return self.forward(args[0],args[1], args[2])

	def parameters(self):
		return self.sc.no_params_func(self.cmd, "params",return_type='FloatTensor_list')
	
	def num_parameters(self):
		return self.sc.no_params_func(self.cmd,"param_count",return_type='int')

	def models(self):
		return self.sc.no_params_func(self.cmd, "models",return_type='Model_list')

	def fit(self, input, target, criterion, optim, iters=15):
		return self.sc.params_func(self.cmd,"fit",[input.id, target.id, criterion.id, optim.id, iters], return_type='FloatTensor')	

	def summary(self, verbose=True, return_instead_of_print = False):

		layer_type = self.layer_type() + "_" + str(self.id) + " (" + str(type(self)).split("'")[1].split(".")[-1] + ")"

		if(type(self.output_shape) == int):
			output_shape = str((None,self.output_shape))
		else:
			output_shape = str(self.output_shape)
			
		n_param = str(self.num_parameters())
		output = layer_type + " "*(29-len(layer_type)) + output_shape + " "*(26-len(output_shape)) + n_param + "\n"
		if(verbose):
			single = "_________________________________________________________________\n"
			header = "Layer (type)                 Output Shape              Param #   \n"
			double = "=================================================================\n"
			total_params = "Total params: " + "{:,}".format(self.num_parameters()) + "\n"
			trainable_params = "Trainable params: " + "{:,}".format(self.num_parameters()) + "\n"
			non_trainable_params = "Non-trainable params: 0" + "\n"
			output = single + header + double + output + double + total_params + trainable_params + non_trainable_params + single

		if(return_instead_of_print):
			return output
		print(output)

	def __len__(self):
		return len(self.models())

	def __getitem__(self,idx):
		return self.parameters()[idx]		

	def activation(self):
		return self.sc.no_params_func(self.cmd, "activation",return_type='FloatTensor')		

	def layer_type(self):
		return self.sc.no_params_func(self.cmd,"model_type",return_type='string')

	def cmd(self,function_call, params = []):
		cmd = {
		'functionCall': function_call,
		'objectType': self.type,
		'objectIndex': self.id,
		'tensorIndexParams': params}
		return cmd

	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')	

	def __repr__(self,verbose=True):

		if(verbose):
			output = ""
			output += self.__repr__(False) + "\n"
			for p in self.parameters():
				output += "\t W:" + p.__repr__(verbose=False)
			activation = self.activation()
			if(activation is not None):
				output += "\t A:" + activation.__repr__(verbose=False) + "\n"

			return output
		else:
			return "<syft.nn."+self._layer_type+" at " + str(self.id) + ">"

# class Policy(Model):
# 	super(Policy, self).__init__()

# 	def __init__(self, model, state_type='discrete'):
		
# 		self.init("policy",[model.id])
# 		self.model = model
# 		self.state_type = state_type

# 	def sample(self, input):
# 		return self.sc.params_func(self.cmd,"sample",[input.id],return_type='IntTensor')	

# 	def __call__(self,*args):

# 		if(self.state_type == 'discrete'):
# 			if(len(args) == 1):
# 				return self.sample(args[0])
# 			elif(len(args) == 2):
# 				return self.sample(args[0],args[1])
# 			elif(len(args) == 3):
# 				return self.sample(args[0],args[1], args[2])

# 		elif(self.state_type == 'continuous'):
# 			if(len(args) == 1):
# 				return self.forward(args[0])
# 			elif(len(args) == 2):
# 				return self.forward(args[0],args[1])
# 			elif(len(args) == 3):
# 				return self.forward(args[0],args[1], args[2])

# 		else:
# 			print("Error: State type " + self.state_type + " unknown")

class Sequential(Model):

	def __init__(self, layers=None):
		super(Sequential, self).__init__()
		
		self.init("sequential")

		if(layers is not None):
			for layer in layers:
				self.add(layer)

	def add(self, model):
		self.sc.params_func(self.cmd,"add",[model.id])

	def summary(self):
		single = "_________________________________________________________________\n"
		header = "Layer (type)                 Output Shape              Param #   \n"
		double = "=================================================================\n"
		total_params = "Total params: " + "{:,}".format(self.num_parameters()) + "\n"
		trainable_params = "Trainable params: " + "{:,}".format(self.num_parameters()) + "\n"
		non_trainable_params = "Non-trainable params: 0" + "\n"

		output = single + header + double

		mods = self.models()

		for m in mods[:-1]:
			output += m.summary(verbose=False, return_instead_of_print=True)

			output += single

		output += mods[-1].summary(verbose=False, return_instead_of_print=True)
		output += double
		output += total_params + trainable_params + non_trainable_params + single
		print(output)
		

	def __repr__(self):
		output = ""
		for m in self.models():
			output += m.__repr__()
		return output

	def __getitem__(self,idx):
		return self.models()[idx]		

class Linear(Model):

	def __init__(self, input_dim=0, output_dim=0, id=None):
		super(Linear, self).__init__()
	
		if(id is None):
			self.init("linear",[input_dim,output_dim])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "linear"

		params = self.parameters()

		self.output_shape = int(params[0].shape()[-1])
		self.input_shape = int(params[0].shape()[0])

class ReLU(Model):
	def __init__(self, id=None):
		super(ReLU, self).__init__()

		if(id is None):
			self.init("relu")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "relu"

class Dropout(Model):
	def __init__(self, rate=0.5, id=None):
		super(Dropout, self).__init__()

		if(id is None):
			self.init("dropout",params=[rate])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "dropout"			

class Sigmoid(Model):
	def __init__(self, id=None):
		super(Sigmoid, self).__init__()

		if(id is None):
			self.init("sigmoid")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "sigmoid"

class Softmax(Model):
	def __init__(self, dim=1, id=None):
		super(Softmax, self).__init__()

		if(id is None):
			self.init("softmax",params=[dim])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "softmax"			

class LogSoftmax(Model):
	def __init__(self, dim=1, id=None):
		super(LogSoftmax, self).__init__()

		if(id is None):
			self.init("logsoftmax",params=[dim])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "logsoftmax"	

class Log(Model):
	def __init__(self, id=None):
		super(Log, self).__init__()

		if(id is None):
			self.init("log")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "log"	

class Tanh(Model):
	def __init__(self, id=None):
		super(Tanh, self).__init__()

		if(id is None):
			self.init("tanh")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "tanh"


class MSELoss(Model):
	def __init__(self, id=None):
		super(MSELoss, self).__init__()

		if (id is None):
			self.init("mseloss")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "mseloss"

	def forward(self, input, target):
		return self.sc.params_func(self.cmd, "forward", [input.id, target.id], return_type='FloatTensor')

class NLLLoss(Model):
	def __init__(self, id=None):
		super(NLLLoss, self).__init__()

		if (id is None):
			self.init("nllloss")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "nllloss"

	def forward(self, input, target):
		return self.sc.params_func(self.cmd, "forward", [input.id, target.id], return_type='FloatTensor')


class CrossEntropyLoss(Model):

	# TODO backward() to be implemented: grad = target - prediction
	# TODO backward(): until IntegerTensor is available assume a one-hot vector is passed in.

	def __init__(self, dim=1, id=None):
		super(CrossEntropyLoss, self).__init__()

		if(id is None):
			self.init("crossentropyloss",params=[dim])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "crossentropyloss"

	def forward(self, input, target):
		return self.sc.params_func(self.cmd, "forward", [input.id, target.id], return_type='FloatTensor')


