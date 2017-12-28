class Variable(object):

	def __init__(self, data):
		self.data = data
		self.data.syft_obj.autograd(True)

	def softmax(self,dim=1):
		return Variable(self.data.softmax())

	def __repr__(self):
		return "Variable:\n" + self.data.__repr__()