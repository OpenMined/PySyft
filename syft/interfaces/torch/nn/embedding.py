import syft
import syft.nn as nn
from syft.interfaces.torch import actual_torch
from syft.interfaces.torch.nn import Module
from syft.interfaces.torch.tensor import FloatTensor

class Embedding(Module):

	def __init__(self, num_embeddings, embedding_dim, syft_obj=None, use_torch_init = True):
		super(Embedding, self).__init__()

		if(syft_obj is None):
			self.syft_obj = nn.Embedding(num_embeddings, embedding_dim)
			if(use_torch_init):
				self.torch_obj = None
		else:
			self.syft_obj = syft_obj


	def parameters(self):
		return map(lambda x:FloatTensor(syft_obj=x),self.syft_obj.parameters())

	def __call__(self,input):
		return syft.interfaces.torch.autograd.Variable(syft.interfaces.torch.FloatTensor(syft_obj=self.syft_obj(input.data.syft_obj)))
