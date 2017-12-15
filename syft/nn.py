class Model():
	def __init__(self,sc):
		self.id=-1
		self.sc = sc
		self.params = False

	def __call__(self,*args):
		if(len(args) == 1):
			return self.forward(args[0])
		elif(len(args) == 2):
			return self.forward(args[0],args[1])
		elif(len(args) == 3):
			return self.forward(args[0],args[1], args[2])

	def parameters(self):
		params = list()
		for v in self.__dict__.values():
		    if(isinstance(v,Model)):
		       for p in v.parameters():
		           params.append(p)
		return params

	def cmd(self,function_call, params = []):
		cmd = {
			'functionCall': function_call,
			'objectType': 'model',
			'objectIndex': self.id,
			'tensorIndexParams': params}
		return cmd

class _Conv(Model):

    def __init__(self, input, output, kernel, stride,
                 padding, dilation, groups, bias):
        super(_Conv, self).__init__(sc)
        if input % groups != 0:
            raise ValueError('input must be divisible by groups')
        if output % groups != 0:
            raise ValueError('output must be divisible by groups')

        assert type(input) == int
        assert type(output) == int
        assert type(kernel) == tuple
        assert type(stride) == tuple
        assert type(padding) == tuple
        assert type(dilation) == tuple
        assert type(groups) == int

        self.input = input
        self.output = output
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

	def __init__(self, sc, input, output, kernel, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
		if(type(kernel) == int):
			kernel = [kernel,kernel]
		if(type(stride) == int):
			kernel = [stride,stride]
		if(type(padding) == int):
			kernel = [padding,padding]
		if(type(dilation) == int):
			dilation = [dilation,dilation]

		assert len(kernel) == 2

		super(Conv2d, self).__init__(sc, 
			input, output, kernel, stride, padding, dilation,
			groups, bias)
		
		if(bias):
			bias = 1
		else:
			bias = 0

		self.sc.socket.send_json(self.cmd("create",["conv2d",
                                                    input,output,
                                                    kernel[0], kernel[1],
                                                    stride[0], stride[1],
                                                    padding[0], padding[1],
                                                    dilation[0], dilation[1],
                                                    bias
                                                    ]))
		self.id = int(self.sc.socket.recv_string())

	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')

	def parameters(self):
		return self.sc.no_params_func(self.cmd, "params",return_type='FloatTensor_list')

class Sequential(Model):

	def __init__(self,sc):
		super(Sequential, self).__init__(sc)
		self.sc.socket.send_json(self.cmd("create",["sequential"]))
		self.id = int(self.sc.socket.recv_string())

	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')

	def add(self, model):
		self.sc.params_func(self.cmd,"add",[model.id])

	def parameters(self):
		return self.sc.no_params_func(self.cmd, "params",return_type='FloatTensor_list')		

class Linear(Model):

	def __init__(self, sc, dims):
		super(Linear, self).__init__(sc)
		assert len(dims) == 2 and type(dims) == tuple

		self.sc.socket.send_json(self.cmd("create",["linear",dims[0],dims[1]]))
		self.id = int(self.sc.socket.recv_string())

	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')

	def parameters(self):
		return self.sc.no_params_func(self.cmd, "params",return_type='FloatTensor_list')

class Sigmoid(Model):

	def __init__(self, sc):
		super(Sigmoid, self).__init__(sc)
		self.sc.socket.send_json(self.cmd("create",["sigmoid"]))
		self.id = int(self.sc.socket.recv_string())

	def forward(self, input):
		return self.sc.params_func(self.cmd,"forward",[input.id],return_type='FloatTensor')


class MSELoss(Model):

	def forward(self, input, target):
		return (input - target) ** 2
