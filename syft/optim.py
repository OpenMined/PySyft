import syft.controller as controller


class Optimizer(object):
	"""
	Base class for all Optimizers to inherit from
	"""

	def __init__(self, id=None):
		self.sc = controller
		self.params = False
		self._optimizer_type = None
		self.id = id
		self.type = "Optimizer"

	def init(self, optimizer_type, params=[], h_params=[]):
		self._optimizer_type = optimizer_type
		self.id = -1
		self.id = int(self.sc.send_json(self.cmd("create", [self._optimizer_type] + params, h_params)))

	def zero_grad(self):
		return self.sc.no_params_func(self.cmd, "zero_grad", return_type='string')

	def step(self, batch_size, iteration):
		return self.sc.params_func(self.cmd, "step", params=[batch_size, iteration], return_type='string')	

	def cmd(self, function_call, params=[], h_params=[]):
		cmd = {
	    'functionCall': function_call,
	    'objectType': self.type,
	    'objectIndex': self.id,
	    'tensorIndexParams': params,
			'hyperParams': h_params}
		return cmd

	def get_param_ids(self, params=[]):
		param_ids = []
		for p in params:
			param_ids.append(p.id)
		return param_ids


class SGD(Optimizer):
	"""
	Stochastic Gradient Descent optimizer.
	Includes support for momentum and learning rate decay
	"""
	def __init__(self, params, lr=0.01, momentum=0., decay=0.):
		super(SGD, self).__init__()
		self.init('sgd', params=self.get_param_ids(params), h_params=[lr, momentum, decay])


class RMSProp(Optimizer):
	"""
	RMSProp Optimizer
	"""
	def __init__(self, params, lr=0.01, rho=0.9, epsilon=1e-8, decay=0.):
		super(RMSProp, self).__init__()
		self.init('rmsprop', params=self.get_param_ids(params), h_params=[lr, rho, epsilon, decay])


class Adam(Optimizer):
	"""
	Adam Optimizer
	"""
	def __init__(self, params, lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.):
		super(Adam, self).__init__()
		self.init('adam', params=self.get_param_ids(params), h_params=[lr, beta_1, beta_2, epsilon, decay])
