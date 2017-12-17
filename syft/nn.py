import syft.controller as controller

class Model():
	def __init__(self,id=None):
		self.id=-1
		self.sc = controller
		self.params = False
		self.type = None
		self._layer_type = None
		self.id = id
		self.type = "model"

	def init(self,layer_type,params=[]):
		self.type = "model"
		self._layer_type = layer_type
		self.sc = controller
		self.id = -1
		self.id = int(self.sc.send_json(self.cmd("create",[self._layer_type] + params)))

	def discover(self):
		self._layer_type = self.layer_type()
		if(self._layer_type == 'linear'):
			return Linear(id = self.id)
		elif(self._layer_type == 'sigmoid'):
			return Sigmoid(id = self.id)

	def __call__(self,*args):
		if(len(args) == 1):
			return self.forward(args[0])
		elif(len(args) == 2):
			return self.forward(args[0],args[1])
		elif(len(args) == 3):
			return self.forward(args[0],args[1], args[2])

	def parameters(self):
		return self.sc.no_params_func(self.cmd, "params",return_type='FloatTensor_list')

	def models(self):
		return self.sc.no_params_func(self.cmd, "models",return_type='Model_list')		

	def layer_type(self):
		return self.sc.no_params_func(self.cmd,"layer_type",return_type='string')

	def cmd(self,function_call, params = []):
		cmd = {
			'functionCall': function_call,
			'objectType': 'model',
			'objectIndex': self.id,
			'tensorIndexParams': params}
		return cmd
    
	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')		

class Sequential(Model):
	def __init__(self, layers=None):		
		self.init("sequential")

		if(layers is not None):
			for layer in layers:
				self.add(layer)

	def add(self, model):
		self.sc.params_func(self.cmd,"add",[model.id])

class Linear(Model):
	def __init__(self, input_dim=0, output_dim=0, id=None):	
		if(id is None):
			self.init("linear",[input_dim,output_dim])
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "linear"

class Sigmoid(Model):
	def __init__(self, id=None):
		if(id is None):
			self.init("sigmoid")
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "sigmoid"

class MSELoss(Model):
	def forward(self, data, target):
		return (data - target) ** 2

class _Conv(Model):
    def __init__(self, input_dim, output_dim, kernel, stride,
                 padding, dilation, groups, bias):
        #super(_Conv, self).__init__(sc)
        if input_dim % groups != 0:
            raise ValueError('input must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output must be divisible by groups')

        assert type(input_dim) == int
        assert type(output_dim) == int
        assert type(kernel) == tuple
        assert type(stride) == tuple
        assert type(padding) == tuple
        assert type(dilation) == tuple
        assert type(groups) == int

        self.input = input_dim
        self.output = output_dim
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __repr__(self):
        s = ('{name} ({input}, {output}, kernel={kernel}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_Conv):
	r"""Applies a 2D convolution/cross-correlation using a kernel.

	| :attr:`stride` shift size of filter motion.
	| :attr:`padding` size of input enlargement with zeros on both sides.
	| :attr:`dilation` inflates filter with zeros like a chess board.
	| :attr:`groups` controls the connections between inputs and outputs.
		`input` and `outout` must both be divisible by `groups`.

	Args:
		input (int): Number of input channels
		output (int): Number of output channels
		kernel (int or tuple): Kernel size
		stride (int or tuple, optional): Stride of the convolution. Default: 1
		padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
		dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
		groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
		bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

	.. _cross-correlation:
		https://en.wikipedia.org/wiki/Cross-correlation

	.. _link:
		https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
	"""
	def __init__(self, input_dim, output_dim, kernel, stride=1,
                padding=0, dilation=1, groups=1, bias=True,id=None):
		if(type(kernel) == int):
			kernel = [kernel,kernel]
		if(type(stride) == int):
			kernel = [stride,stride]
		if(type(padding) == int):
			kernel = [padding,padding]
		if(type(dilation) == int):
			dilation = [dilation,dilation]

		assert len(kernel) == 2

		super(Conv2d, self).__init__(
			input_dim, output_dim, kernel, stride, padding, dilation,
			groups, bias)

		if(bias):
			bias = 1
		else:
			bias = 0

		params = [input_dim,output_dim,
					kernel[0], kernel[1],
					stride[0], stride[1],
					padding[0], padding[1],
					dilation[0], dilation[1],
					bias
					]

		if(id is None):
			self.init("conv2d",params)
		else:
			self.id = id
			self.sc = controller
			self.type = "model"
			self._layer_type = "conv2d"