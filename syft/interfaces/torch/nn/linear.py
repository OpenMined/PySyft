import syft
import syft.nn as nn
from syft.interfaces.torch import actual_torch
from syft.interfaces.torch.nn import Module
from syft.interfaces.torch.tensor import FloatTensor

class Linear(Module):

	def __init__(self, input=None, output=None, syft_obj=None, use_torch_init = True):
		super(Linear, self).__init__()

		if(syft_obj is None):
			self.syft_obj = nn.Linear(input,output)
			if(use_torch_init):
				self.torch_obj = actual_torch.torch.nn.Linear(input,output)

				om_weights, om_bias = self.syft_obj.parameters()
				om_weights *= 0
				om_bias *= 0

				om_weights += syft.FloatTensor(self.torch_obj.weight.data.t().numpy())
				om_bias += syft.FloatTensor(self.torch_obj.bias.data.view(1,-1).numpy())

				self.torch_obj = None

		else:
			self.syft_obj = syft_obj


	def parameters(self):
		return map(lambda x:FloatTensor(syft_obj=x),self.syft_obj.parameters())

	def __call__(self,input):
		return syft.interfaces.torch.autograd.Variable(syft.interfaces.torch.FloatTensor(syft_obj=self.syft_obj(input.data.syft_obj)))