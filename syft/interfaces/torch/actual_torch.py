import sys
try:
	import torch
except:
	sys.stderr.write('Import warning: PyTorch capabilities not available due to torch module not found on your system\nHow to install PyTorch: http://pytorch.org/')

def manual_seed(seed):
	return torch.manual_seed(seed)

def from_numpy(input):
	return syft.interfaces.torch.FloatTensor(input)