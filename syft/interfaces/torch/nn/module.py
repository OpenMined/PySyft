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