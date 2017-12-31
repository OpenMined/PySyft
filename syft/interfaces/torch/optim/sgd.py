import syft.optim

class SGD(object):

	def __init__(self, params, lr):
		self.syft_obj = syft.optim.SGD(map(lambda x:x.syft_obj,params),lr)

	def zero_grad(self):
		self.syft_obj.zero_grad()

	def step(self):
		self.syft_obj.step()
