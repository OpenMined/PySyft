class RMSprop(object):

	def __init__(self, **hyperparameters):
		self.hyperparameters = hyperparameters

	def init(self, syft_params):
		self.syft = optim.RMSProp(syft_params,**self.hyperparameters)