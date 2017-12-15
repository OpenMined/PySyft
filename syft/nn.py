import syft.controller as controller

class Model():
	def __init__(self):
		self.sc = controller
		self.params = False

	def __call__(self,*args):
		if(len(args) == 1):
			return self.forward(args[0])
		elif(len(args) == 2):
			return self.forward(args[0],args[1])
		elif(len(args) == 3):
			return self.forward(args[0],args[1], args[2])

	def parameters(self):
		params = list()
		for v in self.__dict__.values():
		    if(isinstance(v,Model)):
		       for p in v.parameters():
		           params.append(p)
		return params

class Sequential(Model):

	def __init__(self):
		self.sc = controller
		self.id = -1
		self.sc.socket.send_json(self.cmd("create",["sequential"]))
		self.id = int(self.sc.socket.recv_string())

	def cmd(self,function_call, params = []):
		cmd = {
	    'functionCall': function_call,
	    'objectType': 'model',
	    'objectIndex': self.id,
	    'tensorIndexParams': params}
		return cmd

	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')

	def add(self, model):
		self.sc.params_func(self.cmd,"add",[model.id])

	def parameters(self):
		return self.sc.no_params_func(self.cmd, "params",return_type='FloatTensor_list')


class Linear(Model):

	def __init__(self, *dims):
		self.sc = controller
		assert len(dims) == 2 and type(dims) == tuple

		self.id = -1
		self.sc.socket.send_json(self.cmd("create",["linear",dims[0],dims[1]]))
		self.id = int(self.sc.socket.recv_string())

	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')

	def parameters(self):
		return self.sc.no_params_func(self.cmd, "params",return_type='FloatTensor_list')

	def cmd(self,function_call, params = []):
		cmd = {
	    'functionCall': function_call,
	    'objectType': 'model',
	    'objectIndex': self.id,
	    'tensorIndexParams': params}
		return cmd

class Sigmoid(Model):

	def __init__(self):
		self.sc = controller
		self.id = -1
		self.sc.socket.send_json(self.cmd("create",["sigmoid"]))
		self.id = int(self.sc.socket.recv_string())

	def cmd(self,function_call, params = []):
		cmd = {
	    'functionCall': function_call,
	    'objectType': 'model',
	    'objectIndex': self.id,
	    'tensorIndexParams': params}
		return cmd

	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')


class MSELoss(Model):

	def forward(self, input, target):
		return (input - target) ** 2
