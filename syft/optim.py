import syft.controller as controller

class SGD(object):

	def __init__(self, params, lr=0.01, momentum=0., decay=0.):

		param_ids = list()
		for p in params:
			param_ids.append(p.id)
		self.type = 'Optimizer'
		self.sc = controller
		self.id = -1
		self.id = int(self.sc.send_json(self.cmd("create", [lr, momentum, decay] + param_ids)))

	def zero_grad(self):
		return self.sc.no_params_func(self.cmd,"zero_grad",return_type='string')

	def step(self, batch_size, iteration):
		return self.sc.params_func(self.cmd, "step", params=[batch_size, iteration], return_type='string')		

	def cmd(self,function_call, params = []):
		cmd = {
	    'functionCall': function_call,
	    'objectType': self.type,
	    'objectIndex': self.id,
	    'tensorIndexParams': params}
		return cmd

