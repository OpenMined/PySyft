import syft.optim as optim

class Adam(object):

	def __init__(self, **hyperparameters):
		self.hyperparameters = hyperparameters

	def init(self, syft_params):
		self.syft = optim.Adam(syft_params,**self.hyperparameters)