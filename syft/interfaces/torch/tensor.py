import syft

def Tensor(data):
	return FloatTensor(data=data)

class IntTensor(object):

	def __init__(self,data=None,syft_obj=None):
		if(syft_obj is None):
			self.syft_obj = syft.IntTensor(data)
		else:
			self.syft_obj = syft_obj

	def __getitem__(self,i):
		out = self.syft_obj.to_numpy()[i]
		if(out.shape == ()):
			return int(out)
		else:
			return IntTensor(out)

	def __repr__(self):
		return self.syft_obj.__repr__()

class FloatTensor(object):

	def __init__(self,data=None,syft_obj=None):
		if(syft_obj is None):
			self.syft_obj = syft.FloatTensor(data)
		else:
			self.syft_obj = syft_obj

		self.current = 0
	
	def index_select(self,dim,indices):
		return FloatTensor(syft_obj=self.syft_obj.index_select(dim,indices.syft_obj))

	def softmax(self,dim=-1):
		return FloatTensor(syft_obj=self.syft_obj.softmax())

	def float(self):
		return self

	def sum(self,dim=-1):
		return FloatTensor(syft_obj=self.syft_obj.sum(dim))

	def log(self):
		return FloatTensor(syft_obj=self.syft_obj.log())

	def mean(self, dim=-1):
		return FloatTensor(syft_obj = self.syft_obj.mean(dim))

	def __neg__(self):
		return FloatTensor(syft_obj = self.syft_obj.__neg__())

	def std(self, dim=-1):
		return FloatTensor(syft_obj = self.syft_obj.std(dim))		

	def unsqueeze(self, dim=0):
		return FloatTensor(syft_obj=self.syft_obj.unsqueeze(dim))

	def sample(self, dim = -1):
		return IntTensor(syft_obj=self.syft_obj.sample(dim))

	def __sub__(self,x):
		if(type(x) == type(self)):
			return FloatTensor(syft_obj = self.syft_obj - x.syft_obj)
		else:
			return FloatTensor(syft_obj = self.syft_obj - x)

	def __add__(self,x):
		if(type(x) == type(self)):
			return FloatTensor(syft_obj = self.syft_obj + x.syft_obj)
		else:
			return FloatTensor(syft_obj = self.syft_obj + x)		

	def __mul__(self,x):
		if(type(x) == type(self)):
			return FloatTensor(syft_obj = self.syft_obj * x.syft_obj)
		else:
			return FloatTensor(syft_obj = self.syft_obj * x)		

	def __truediv__(self,x):
		if(type(x) == type(self)):
			return FloatTensor(syft_obj = self.syft_obj / x.syft_obj)
		else:
			return FloatTensor(syft_obj = self.syft_obj / x)	

	def __len__(self):
		return self.syft_obj.shape()[0]

	def __iter__(self):
		self.i = 0
		self.cached_data = self.syft_obj.to_numpy()
		while(self.i < len(self)):
			yield self.cached_data[self.i]
			self.i += 1

	def __repr__(self):
		return self.syft_obj.__repr__()
