class Variable(object):

	def __init__(self, data):
		self.data = data
		self.data.syft_obj.autograd(True)

	def softmax(self,dim=1):
		return Variable(self.data.softmax())

	def sample(self,dim=1):
		return Variable(self.data.sample(dim))

	def index_select(self,dim,indices):
		return Variable(self.data.index_select(dim,indices.data))

	def log(self):
		return Variable(self.data.log())

	def __neg__(self):
		return self.data.__neg__();

	def __repr__(self):
		return "Variable:\n" + self.data.__repr__()