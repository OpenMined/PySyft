import syft.controller

class Model():
	def __init__(self):
		self.controller = syft.controller
		self.controller.log("Model instantiated")

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
		       if(v.params):
		           params.append(v)
		return params

class Linear(Model):
	def __init__(self, dims):
		self.controller.socket.send_json({
			"objectType": "linear",
			"functionCall": "create",
			"dimensions": dims
		})

		self.id = int(self.controller.socket.recv_string())
		if (OpenMinedController.verbose):
			print("Linear.__init__: {}".format(self.id))

	def forward(self, input):
		return input.mm(self.weights)


class Sigmoid(Model):

	def forward(self, input):
		return input.sigmoid();


class MSELoss():

	def __call__(self, *args):
		return self.forward(args[0], args[1])

	def forward(self, input, target):
		return (input - target) ** 2
