import syft

class FloatTensor(object):

	def __init__(self,data=None,syft_obj=None):
		if(syft_obj is None):
			self.syft_obj = syft.FloatTensor(data)
		else:
			self.syft_obj = syft_obj