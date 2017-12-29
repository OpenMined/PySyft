from syft.interfaces.torch import FloatTensor

def softmax(x,dim=1):
	return x.softmax(dim=dim)