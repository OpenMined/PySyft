class Model():
	def __init__(self,sc):
		self.sc = sc
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
		       if(v.params):
		           params.append(v)
		return params

class Linear(Model):

	def __init__(self, sc, dims):
		self.sc = sc
		assert len(dims) == 2 and type(dims) == tuple
		self.dims = dims
		self.params = True

		self.weights = ((sc.randn(self.dims[0],self.dims[1]) * 0.2) - 0.1).autograd(True)
		self.bias = sc.zeros(1,self.dims[1]).autograd(True)

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
