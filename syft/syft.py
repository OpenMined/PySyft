import zmq
import uuid


class FloatTensor():

    def __init__(self, controller, data, data_is_pointer=False, verbose=False):
        self.verbose = verbose
        self.controller = controller
        if(data is not None and not data_is_pointer):
            data = data.astype('float')
            controller.socket.send_json({"objectType": "tensor",
                                         "functionCall": "create",
                                         "data": list(data.flatten()),
                                         "shape": data.shape})
            self.id = int(controller.socket.recv_string())
            if(verbose):
                print("FloatTensor.__init__: " +  str(self.id))

        elif(data_is_pointer):
            self.id = int(data)

    def __del__(self):
        self.delete_tensor()

    def abs(self):
        return self.no_params_func("abs", return_response=True)

    def abs_(self):
        return self.no_params_func("abs_")

    def acos(self):
        return self.no_params_func("acos", return_response=True)

    def acos_(self):
        return self.no_params_func("acos_")

    def asin(self):
        return self.no_params_func("asin", return_response=True)

    def asin_(self):
        return self.no_params_func("asin_")

    def atan(self):
        return self.no_params_func("atan", return_response=True)

    def atan_(self):
        return self.no_params_func("atan_")

    def addmm_(self, x, y):
        return self.params_func("addmm_", [x.id, y.id])

    def addmm(self, x, y):
        copy = self.copy()
        copy.params_func("addmm_", [x.id, y.id])
        return copy

    def addmv_(self, x, y):
        return self.params_func("addmv_", [x.id, y.id])

    def addmv(self, x, y):
        copy = self.copy()
        copy.params_func("addmv_", [x.id, y.id])
        return copy

    def __add__(self, x):
        return self.arithmetic_operation(x, "add", False)

    def __iadd__(self, x):
        return self.arithmetic_operation(x, "add", True)

    def copy(self):
        return self.no_params_func("copy", return_response=True)

    def cos(self):
        return self.no_params_func("cos", return_response=True)

    def cos_(self):
        return self.no_params_func("cos_")

    def cosh(self):
        return self.no_params_func("cosh", return_response=True)

    def cosh_(self):
        return self.no_params_func("cosh_")

    def __truediv__(self, x):
        return self.arithmetic_operation(x, "div", False)

    def __itruediv__(self, x):
        return self.arithmetic_operation(x, "div", True)

    def floor_(self):
        return self.no_params_func("floor_")

    def __mul__(self, x):
        return self.arithmetic_operation(x, "mul", False)

    def __imul__(self, x):
        return self.arithmetic_operation(x, "mul", True)

    def neg(self):
        return self.no_params_func("neg", return_response=True)

    def rsqrt(self):
        return self.no_params_func("rsqrt",return_response=True)

    def sigmoid_(self):
        return self.no_params_func("sigmoid_")

    def sign(self):
        return self.no_params_func("sign", return_response=True)

    def sign_(self):
        return self.no_params_func("sign_")

    def sin(self):
        return self.no_params_func("sin", return_response=True)

    def sin_(self):
        return self.no_params_func("sin_")

    def size(self):
        """
        Returns the size of the self tensor as a FloatTensor.

        Note:
            The returned value currently is a FloatTensor because it leverages
            the messaging mechanism with Unity.
        """
        return self.no_params_func("size", return_response=True)

    def sqrt(self):
        return self.no_params_func("sqrt", return_response=True)

    def trace(self):
        return self.no_params_func("trace", return_response=True)

    def trunc(self):
        return self.no_params_func("trunc", return_response=True)

    def __sub__(self, x):
        return self.arithmetic_operation(x, "sub", False)

    def __isub__(self,x):
        return self.arithmetic_operation(x,"sub",True)

    def sum(self,dim):
        assert type(dim) == int
        return self.arithmetic_operation(dim, "sum", False)

    def view(self, *args):
        new_dim = list(args)
        assert type(new_dim) == list
        assert type(new_dim[0]) == int
        return self.params_func("view", new_dim, return_response=True)

    def view_(self, *args):
        new_dim = list(args)
        assert type(new_dim) == list
        assert type(new_dim[0]) == int
        self.params_func("view_", new_dim, return_response=False)
        return self

    def T(self):
        return self.no_params_func("transpose", return_response=True)

    def triu(self, k=0):
        return self.params_func("triu", [k], return_response=True)

    def triu_(self, k=0):
        return self.params_func("triu_", [k])

    # Fills this tensor with zeros.
    def zero_(self):
        return self.no_params_func("zero_")

    def __repr__(self):
        return self.no_params_func("print", True, False)

    def __str__(self):
        return self.no_params_func("print", True, False)

    def cpu(self):
        return self.no_params_func("cpu")

    def gpu(self):
        return self.no_params_func("gpu")

    def cmd(self, functionCall, tensorIndexParams=[]):
        cmd = {
            'functionCall': functionCall,
            'objectType': 'tensor',
            'objectIndex': self.id,
            'tensorIndexParams': tensorIndexParams}
        return cmd

    def params_func(self, name, params, return_response=False, return_as_tensor=True):
        # send the command
        self.controller.socket.send_json(
            self.cmd(name, tensorIndexParams=params))
        # receive output from command
        res = self.controller.socket.recv_string()

        if(self.verbose):
            print(res)

        if(return_response):
            if(return_as_tensor):
                if(self.verbose):
                    print("FloatTensor.__init__: " +  res)
                return FloatTensor(self.controller,int(res),True)
            else:
                return res
        return self

    def no_params_func(self, name, return_response=False, return_as_tensor=True):
        return(self.params_func(name, [], return_response, return_as_tensor))

    def arithmetic_operation(self, x, name, inline=False):

        operation_cmd = name

        if(type(x) == FloatTensor):
            operation_cmd += "_elem"
            parameter = x.id
        else:
            operation_cmd += "_scalar"
            parameter = str(x)

        if(inline):
            operation_cmd += "_"

        self.controller.socket.send_json(
            self.cmd(operation_cmd, [parameter]))  # sends the command
        return FloatTensor(self.controller, int(self.controller.socket.recv_string()), True)

    def delete_tensor(self):
        if(self.id is not None):
            self.no_params_func("delete")
        self.verbose = None
        self.controller = None
        self.id = None

    def T(self):
        return self.no_params_func("transpose", return_response=True)

    def is_contiguous(self):
        return self.no_params_func("is_contiguous", return_response=True, return_as_tensor=False)

    def sinh(self):
        return self.no_params_func("sinh", return_response=True)

    def sinh_(self):
        return self.no_params_func("sinh_")

    def tan(self):
        return self.no_params_func("tan", return_response=True)

    def tan_(self):
        return self.no_params_func("tan_")

    def tanh(self):
        return self.no_params_func("tanh", return_response=True)


class SyftController():

    def __init__(self,verbose=True):

        self.identity = str(uuid.uuid4())

        context = zmq.Context()
        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.identity)
        self.socket.connect("tcp://localhost:5555")
        self.verbose=verbose

    def FloatTensor(self, data):
        verbose = self.verbose
        return FloatTensor(self, data,verbose=verbose)
