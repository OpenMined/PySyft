from syft.interfaces.torch.tensor import FloatTensor
from syft.interfaces.torch.autograd import Variable
import syft

def softmax(x,dim=1):
	return x.softmax(dim=dim)

def cat(tensors,axis=0):
	if(type(tensors[0]) == FloatTensor):
		return FloatTensor(syft_obj=syft.concatenate(list(map(lambda x:x.syft_obj,tensors)),axis=axis))
	elif(type(tensors[0]) == Variable):
		return Variable(cat(tensors=list(map(lambda x:x.data,tensors)),axis=axis))
	else:
		print("Not supported for type: " + str(type(tensors[0])))
