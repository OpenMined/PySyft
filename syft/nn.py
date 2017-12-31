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
		elif(self._layer_type == 'crossentropyloss'):
			return CrossEntropyLoss(id = self.id)
		elif(self._layer_type == 'tanh'):
			return Tanh(id = self.id)
		elif(self._layer_type == 'dropout'):
			return Dropout(id = self.id)

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

	def fit(self, input, target, criterion, optim, iters=15):
		return self.sc.params_func(self.cmd,"fit",[input.id, target.id, criterion.id, optim.id, iters], return_type='FloatTensor')	

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

class Policy(Model):

	def __init__(self, model, state_type='discrete'):
		
		self.init("policy",[model.id])
		self.model = model
		self.state_type = state_type

	def sample(self, input):
		return self.sc.params_func(self.cmd,"sample",[input.id],return_type='IntTensor')	

	def __call__(self,*args):

		if(self.state_type == 'discrete'):
			if(len(args) == 1):
				return self.sample(args[0])
			elif(len(args) == 2):
				return self.sample(args[0],args[1])
			elif(len(args) == 3):
				return self.sample(args[0],args[1], args[2])

		elif(self.state_type == 'continuous'):
			if(len(args) == 1):
				return self.forward(args[0])
			elif(len(args) == 2):
				return self.forward(args[0],args[1])
			elif(len(args) == 3):
				return self.forward(args[0],args[1], args[2])

		else:
			print("Error: State type " + self.state_type + " unknown")

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

	def __getitem__(self,idx):
		return self.models()[idx]		

class Linear(Model):

	def __init__(self, input_dim=0, output_dim=0, id=None):
	
		if(id is None):
			self.init("linear",[input_dim,output_dim])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "linear"

class ReLU(Model):
	def __init__(self, id=None):
		if(id is None):
			self.init("relu")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "relu"

class Dropout(Model):
	def __init__(self, rate=0.5, id=None):
		if(id is None):
			self.init("dropout",params=[rate])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "dropout"			

class Sigmoid(Model):
	def __init__(self, id=None):
		if(id is None):
			self.init("sigmoid")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "sigmoid"

class Tanh(Model):
    def __init__(self, id=None):
        if(id is None):
            self.init("tanh")
        else:
            self.id = id
            self.sc = controller
            self.type = "model"
            self._layer_type = "tanh"


class MSELoss(Model):
    def __init__(self, id=None):
        if (id is None):
            self.init("mseloss")
        else:
            self.id = id
            self.sc = controller
            self.type = "model"
            self._layer_type = "mseloss"

    def forward(self, input, target):
        return self.sc.params_func(self.cmd, "forward", [input.id, target.id], return_type='FloatTensor')

class CrossEntropyLoss(Model):
    # TODO backward() to be implemented: grad = target - prediction
    # TODO backward(): until IntegerTensor is available assume a one-hot vector is passed in.

    def __init__(self, id=None):
        if(id is None):
            self.init("crossentropyloss")
        else:
            self.id = id
            self.sc = controller
            self.type = "model"
            self._layer_type = "crossentropyloss"

    def forward(self, input, target):
        return self.sc.params_func(self.cmd, "forward", [input.id, target.id], return_type='FloatTensor')


