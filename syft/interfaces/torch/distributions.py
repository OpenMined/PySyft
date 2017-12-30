class Categorical(object):

	def __init__(self, probs):
		self.probs = probs

	def sample(self):
		return self.probs.sample(1)

	def log_prob(self, action):
		return self.probs.index_select(1,action).log()