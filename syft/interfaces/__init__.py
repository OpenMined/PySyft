import sys
try:
	import torch
except:
	sys.stderr.write('Import warning: PyTorch capabilities not available due to torch module not found on your system\nHow to install PyTorch: http://pytorch.org/')