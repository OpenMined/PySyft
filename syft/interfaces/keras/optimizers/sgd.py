import syft.optim as optim

class SGD(object):

	def __init__(self, **hyperparameters):
		self.hyperparameters = hyperparameters

	def init(self, syft_params):
		self.syft = optim.SGD(syft_params,**self.hyperparameters)