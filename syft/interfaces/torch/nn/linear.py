import syft
import syft.nn as nn

class Linear(object):

	def __init__(self, input=None, output=None, syft_obj=None):
		if(syft_obj is None):
			self.syft_obj = nn.Linear(input,output)
		else:
			self.syft_obj = syft_obj


	def __call__(self,input):
		return syft.interfaces.torch.FloatTensor(syft_obj=self.syft_obj(input.syft_obj))