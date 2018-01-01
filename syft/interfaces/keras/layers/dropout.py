class Dropout(object):

	def __init__(self, rate):
		self.rate = rate
		self.ordered_syft = []

	def create_model(self):
		"do nothing"