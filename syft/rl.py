import syft.controller as controller

class Agent():

	def __init__(self, model, optimizer, state_type='discrete'):
		
		self.init([model.id, optimizer.id])
		self.model = model
		self.state_type = state_type
		self.optimizer = optimizer

	def init(self,params=[]):
		self.type = "agent"
		self.sc = controller
		self.id = -1
		self.id = int(self.sc.send_json(self.cmd("create",params)))

	def sample(self, input):
		return self.sc.params_func(self.cmd,"sample",[input.id],return_type='IntTensor')	

	def deploy(self):
		self.int = self.sc.no_params_func(self.cmd,"deploy",return_type="int")

	def cmd(self,function_call, params = []):
		cmd = {
		'functionCall': function_call,
		'objectType': self.type,
		'objectIndex': self.id,
		'tensorIndexParams': params}
		return cmd

	def parameters(self):
		return self.model.parameters()

	def __call__(self,*args):

		if(self.state_type == 'discrete'):
			if(len(args) == 1):
				return self.sample(args[0])
			elif(len(args) == 2):
				return self.sample(args[0],args[1])
			elif(len(args) == 3):
				return self.sample(args[0],args[1], args[2])

		elif(self.state_type == 'continuous'):
			if(len(args) == 1):
				return self.forward(args[0])
			elif(len(args) == 2):
				return self.forward(args[0],args[1])
			elif(len(args) == 3):
				return self.forward(args[0],args[1], args[2])

		else:
			print("Error: State type " + self.state_type + " unknown")

	def history(self):

		raw_history = self.sc.params_func(self.cmd,"get_history",[],return_type="string")

		if(raw_history == ""):
			return [],[]

		history_idx = list(map(lambda x:list(map(lambda y:int(y),x.split(","))),raw_history[2:-1].split("],[")))
		losses = list()
		rewards = list()

		for loss,reward in history_idx:
			if(loss != -1):
				losses.append(self.sc.get_tensor(loss))
			else:
				losses.append(None)
			if(reward != -1):
				rewards.append(self.sc.get_tensor(reward))
			else:
				rewards.append(None)

		return losses,rewards