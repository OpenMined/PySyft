class Module(object):

	def __init__(self):
		""

	def __call__(self,*args):
		if(len(args) == 1):
			return self.forward(args[0])
		elif(len(args) == 2):
			return self.forward(args[0],args[1])
		elif(len(args) == 3):
			return self.forward(args[0],args[1], args[2])

	def parameters(self):
		params = list()

		for item in self.__dict__.items():
		    if(isinstance(item[1],Module)):
		        for p in item[1].parameters():
		            params.append(p)

		return params