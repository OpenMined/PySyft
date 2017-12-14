import zmq
import uuid
import numpy as np
from .nn import Linear, Sigmoid, Sequential

class FloatTensor():

    def __init__(self, controller, data, autograd=False, data_is_pointer=False, verbose=False):
        self.verbose = verbose
        self.controller = controller

        if(data is not None and not data_is_pointer):
            if(type(data) == list):
                data = np.array(data)
            data = data.astype('float')

            self.data = data
            controller.socket.send_json({"objectType": "tensor",
                                         "functionCall": "create",
                                         "data": list(data.flatten()),
                                         "shape": self.data.shape})
            self.id = int(controller.socket.recv_string())
            if(verbose):
                print("FloatTensor.__init__: " +  str(self.id))

        elif(data_is_pointer):
            self.id = int(data)

        if(autograd):
            self.autograd(True)

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

    def asin(self):
        return self.no_params_func("asin", return_response=True)

    def asin_(self):
        return self.no_params_func("asin_")

    def atan(self):
        return self.no_params_func("atan", return_response=True)

    def atan_(self):
        return self.no_params_func("atan_")

    def autograd(self, setter=None):
        if(setter is None):
            if(self.get("autograd") == "1"):
                return True
            else:
                return False
        else:
            if(setter):
                out = self.set("autograd",["1"])
            else:
                out = self.set("autograd",["0"])

            if(out == "1" and setter) or (out == "0" and not setter):
                return self
            else:
                return False

    def __add__(self, x):
        return self.arithmetic_operation(x, "add", False)

    def __iadd__(self, x):
        return self.arithmetic_operation(x, "add", True)

    def backward(self, grad=None):
        if(grad is None):
            self.no_params_func("backward")
        else:
            self.params_func(name="backward",params=[grad.id])

    def ceil(self):
        return self.no_params_func("ceil", return_response=True)

    def ceil_(self):
        return self.no_params_func("ceil_")

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

    def children(self):
        res = self.get("children")
        if(len(res) > 0):
            return list(map(lambda x:int(x),res.split(",")[0:-1]))
        return []

    def creation_op(self):
        return self.get("creation_op")

    def creators(self):
        res = self.get("creators")
        if(len(res) > 0):
            return list(map(lambda x:int(x),res.split(",")[0:-1]))
        return []

    def dataOnGpu(self):
        if(self.get("dataOnGpu") == "1"):
            return True
        return False

    def exp(self):
        return self.no_params_func("exp", return_response=True)

    def exp_(self):
        return self.no_params_func("exp_")

    def __truediv__(self, x):
        return self.arithmetic_operation(x, "div", False)

    def __itruediv__(self, x):
        return self.arithmetic_operation(x, "div", True)

    def keepgrad(self):
        if(self.get("keepgrad") == "1"):
            return True
        else:
            return False

    def __pow__(self, x):
        return self.arithmetic_operation(x, "pow", False)

    def __ipow__(self, x):
        return self.arithmetic_operation(x, "pow", True)

    def pow(self, x):
        return self.arithmetic_operation(x, "pow", False)

    def pow_(self, x):
        return self.arithmetic_operation(x, "pow", True)

    def floor(self):
        return self.no_params_func("floor", True)

    def floor_(self):
        return self.no_params_func("floor_")

    def round(self):
        return self.no_params_func("round", return_response=True)

    def round_(self):
        return self.no_params_func("round_")

    def mm(self, other):
        return self.params_func("mm",[other.id],True)

    def grad(self):
        return self.get("grad", response_as_tensor=True)

    def __mod__(self, x):
        return self.arithmetic_operation(x, "remainder", False)

    def __imod__(self, x):
        return self.arithmetic_operation(x, "remainder", True)

    def __mul__(self, x):
        return self.arithmetic_operation(x, "mul", False)

    def __imul__(self, x):
        return self.arithmetic_operation(x, "mul", True)

    def neg(self):
        return self.no_params_func("neg", return_response=True)

    def neg_(self):
        return self.no_params_func("neg_")

    def rsqrt(self):
        return self.no_params_func("rsqrt",return_response=True)

    def set(self, param_name="size", params=[]):
        return self.params_func(name="set",params=[param_name] + params, return_response=True, return_as_tensor=False)

    def sigmoid_(self):
        return self.no_params_func("sigmoid_")

    def sigmoid(self):
        return self.no_params_func("sigmoid", return_response=True)

    def sign(self):
        return self.no_params_func("sign", return_response=True)

    def sign_(self):
        return self.no_params_func("sign_")

    def sin(self):
        return self.no_params_func("sin", return_response=True)

    def sin_(self):
        return self.no_params_func("sin_")

    def size(self):
        return int(self.get("size"))

    def shape(self,as_list=True):
        """
        Returns the size of the self tensor as a FloatTensor.

        Note:
            The returned value currently is a FloatTensor because it leverages
            the messaging mechanism with Unity.
        """
        shape_tensor = self.no_params_func("shape", return_response=True)
        if(as_list):
            return list(map(lambda x:int(x),shape_tensor.get("data").split(",")[:-1]))
        return shape_tensor

    def stride(self, dim=-1):
        if dim == -1:
            return self.no_params_func("stride", return_response=True, return_as_tensor=False)
        else:
            strides = self.params_func("stride", [dim], return_response=True, return_as_tensor=False)
            return np.fromstring(strides, sep=' ').astype('long')

    def sqrt(self):
        return self.no_params_func("sqrt", return_response=True)

    def trace(self):
        return self.no_params_func("trace", return_response=True)

    def trunc(self):
        return self.no_params_func("trunc", return_response=True)

    def to_numpy(self):
         self.controller.socket.send_json({
             'functionCall': 'to_numpy',
             'objectType': 'tensor',
             'objectIndex': self.id
         })

         res = self.controller.socket.recv_string()
         return np.fromstring(res, sep=' ').astype('float').reshape(self.shape())

    def __sub__(self, x):
        return self.arithmetic_operation(x, "sub", False)

    def __isub__(self,x):
        return self.arithmetic_operation(x,"sub",True)

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

    def view_as(self, x):
        assert type(x) == FloatTensor
        return self.params_func("view_as", [x.id], return_response=True)

    def view_as_(self, x):
        assert type(x) == FloatTensor
        self.params_func("view_as_", [x.id], return_response=False)
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
        tensor_str = str(self.to_numpy())

        type_str = ""
        for dim in self.shape():
            type_str += str(dim) + "x"

        type_str = type_str[:-1]
        return tensor_str + "\n[syft.FloatTensor of size " + type_str + "]" + "\n"
        # return self.no_params_func("print", True, False)

    def __str__(self):
        tensor_str =  str(self.to_numpy()).replace("]"," ").replace("["," ") + "\n"

    def get(self, param_name="size", response_as_tensor=False):
        return self.params_func(name="get",params=[param_name], return_response=True, return_as_tensor=response_as_tensor)

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
                return FloatTensor(controller=self.controller,data=int(res),data_is_pointer=True)
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
        return FloatTensor(controller=self.controller, data=int(self.controller.socket.recv_string()), data_is_pointer=True)

    def delete_tensor(self):
        if(self.id is not None):
            self.no_params_func("delete")
        self.verbose = None
        self.controller = None
        self.id = None

    def is_contiguous(self):
        return self.no_params_func("is_contiguous", return_response=True, return_as_tensor=False)

    def sinh(self):
        return self.no_params_func("sinh", return_response=True)

    def sinh_(self):
        return self.no_params_func("sinh_")

    def log(self):
        return self.no_params_func("log", return_response=True)

    def log_(self):
        return self.no_params_func("log_")

    def log1p_(self):
        return self.no_params_func("log1p_")

    def log1p(self):
        return self.no_params_func("log1p", return_response=True)

    def frac(self):
        return self.no_params_func("frac", return_response=True)

    def frac_(self):
        return self.no_params_func("frac_")

    def reciprocal(self):
        return self.no_params_func("reciprocal", return_response=True)

    def reciprocal_(self):
        return self.no_params_func("reciprocal_")

    def rsqrt(self):
        return self.no_params_func("rsqrt",return_response=True)

    def rsqrt_(self):
        return self.no_params_func("rsqrt_")

    def remainder(self,divisor):
        return self.arithmetic_operation(divisor, "remainder")

    def remainder_(self,divisor):
        return self.arithmetic_operation(divisor, "remainder",True)

    def tan(self):
        return self.no_params_func("tan", return_response=True)

    def tan_(self):
        return self.no_params_func("tan_")

    def tanh(self):
        return self.no_params_func("tanh", return_response=True)

    def squeeze(self, dim=-1):
        return self.params_func("squeeze", [dim], return_response=True)

    def squeeze_(self, dim=-1):
        return self.params_func("squeeze_", [dim])

    def min(self, dim=-1, keepdim=False):
        return self.params_func("min", [dim, keepdim], return_response=True)

    def max(self, dim=-1, keepdim=False):
        return self.params_func("max", [dim, keepdim], return_response=True)

    def sum(self, dim=-1, keepdim=False):
        return self.params_func("sum", [dim, keepdim], return_response=True)

    def prod(self, dim=-1, keepdim=False):
        return self.params_func("prod", [dim, keepdim], return_response=True)

    def mean(self, dim=-1, keepdim=False):
        return self.params_func("mean", [dim, keepdim], return_response=True)

class SyftController():

    def __init__(self,verbose=True):

        self.identity = str(uuid.uuid4())

        context = zmq.Context()
        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.identity)
        self.socket.connect("tcp://localhost:5555")
        self.verbose=verbose

    def FloatTensor(self, data, autograd=False):
        return FloatTensor(controller=self, data=data, autograd=autograd, verbose=self.verbose)

    def Linear(self, *args):
        return Linear(sc=self, dims = args)

    def Sigmoid(self):
        return Sigmoid(sc=self)

    def Sequential(self):
        return Sequential(sc=self)

    def rand(self, *args):
        return self.FloatTensor(np.random.rand(*args))

    def randn(self, *args):
        return self.FloatTensor(np.random.randn(*args))

    def zeros(self,*args):
        return self.FloatTensor(np.zeros((args)))

    def ones(self,*args):
        return self.FloatTensor(np.ones((args)))

    def params_func(self, cmd_func, name, params, return_type=None):
        # send the command
        self.socket.send_json(
            cmd_func(name, params=params))
        # receive output from command
        res = self.socket.recv_string()

        if(self.verbose):
            print(res)

        if(return_type is None):
            return self
        elif(return_type == 'FloatTensor'):
            if(self.verbose):
                print("FloatTensor.__init__: " +  res)
            return FloatTensor(controller=self,data=int(res),data_is_pointer=True)
        elif return_type == 'FloatTensor_list':
            tensors = list()
            if(res[-1] == ','):
                res = res[:-1]
            for str_id in res.split(","):
                tensors.append(FloatTensor(controller=self,data=int(str_id),data_is_pointer=True))
            return tensors
        else:
            return res

    def no_params_func(self, cmd_func, name, return_type):
        return self.params_func(cmd_func, name, [], return_type)
                
