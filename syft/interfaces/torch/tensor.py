import syft

class FloatTensor(object):

	def __init__(self,data=None,syft_obj=None):
		if(syft_obj is None):
			self.syft_obj = syft.FloatTensor(data)
		else:
			self.syft_obj = syft_obj
	
	def softmax(self,dim=-1):
		return FloatTensor(syft_obj=self.syft_obj.softmax())

	def float(self):
		return self

	def unsqueeze(self, dim=0):
		return FloatTensor(syft_obj=self.syft_obj.unsqueeze(dim))

	def __repr__(self):
		return self.syft_obj.__repr__()
