import syft.optim as optim

class SGD(object):

	def __init__(self):
		""

	def init(self, syft_params, alpha):
		self.syft = optim.SGD(syft_params,alpha)