import syft.controller as controller

class Model():
	def __init__(self, id=None):
		self.sc = controller
		self.params = False
		self.type = None
		self._layer_type = None
		self.id = id
		self.type = "model"

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

	def __call__(self,*args):
		if(len(args) == 1):
			return self.forward(args[0])
		elif(len(args) == 2):
			return self.forward(args[0],args[1])
		elif(len(args) == 3):
			return self.forward(args[0],args[1], args[2])

	def parameters(self):
		return self.sc.no_params_func(self.cmd, "params",return_type='FloatTensor_list')

	def models(self):
		return self.sc.no_params_func(self.cmd, "models",return_type='Model_list')		

	def activation(self):
		return self.sc.no_params_func(self.cmd, "activation",return_type='FloatTensor')		

	def layer_type(self):
		return self.sc.no_params_func(self.cmd,"layer_type",return_type='string')

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


class Sequential(Model):

	def __init__(self, layers=None):
		
		self.init("sequential")

		if(layers is not None):
			for layer in layers:
				self.add(layer)

	def add(self, model):
		self.sc.params_func(self.cmd,"add",[model.id])

	def __repr__(self):
		output = ""
		for m in self.models():
			output += m.__repr__()
		return output

class Linear(Model):

	def __init__(self, input_dim=0, output_dim=0, id=None):
	
		if(id is None):
			self.init("linear",[input_dim,output_dim])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "linear"

class Sigmoid(Model):
	def __init__(self, id=None):
		if(id is None):
			self.init("sigmoid")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "sigmoid"


class MSELoss(Model):

	def forward(self, input, target):
		return (input - target) ** 2
