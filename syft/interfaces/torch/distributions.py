class Categorical(object):

	def __init__(self, probs):
		self.probs = probs

	def sample(self):
		return self.probs.sample(1)