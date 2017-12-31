class Variable(object):

	def __init__(self, data):
		self.data = data
		self.data.syft_obj.autograd(True)

	def backward(self,grad=None):
		if(grad is None):
			self.data.syft_obj.backward()
		else:
			self.data.syft_obj.backward(grad)

	def softmax(self,dim=1):
		return Variable(self.data.softmax())

	def sample(self,dim=1):
		return Variable(self.data.sample(dim))

	def index_select(self,dim,indices):
		return Variable(self.data.index_select(dim,indices.data))

	def log(self):
		return Variable(self.data.log())

	def sum(self,dim=-1):
		return Variable(self.data.sum(dim))

	def __neg__(self):
		return Variable(self.data.__neg__());

	def __repr__(self):
		return "Variable:\n" + self.data.__repr__()

	def __sub__(self,x):
		if(type(x) == type(self)):
			return Variable(self.data - x.data)
		else:
			return Variable(self.data - x)

	def __add__(self,x):
		if(type(x) == type(self)):
			return Variable(self.data + x.data)
		else:
			return Variable(self.data + x)	

	def __mul__(self,x):
		if(type(x) == type(self)):
			return Variable(self.data * x.data)
		else:
			return Variable(self.data * x)		

	def __truediv__(self,x):
		if(type(x) == type(self)):
			return Variable(self.data / x.data)
		else:
			return Variable(self.data / x)