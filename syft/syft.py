import uuid
import json
import zmq
import numpy as np
from .nn import Linear, Sigmoid, Sequential


class FloatTensor():
    """
    FloatTensor is a stub class which basically delegates most calls to the backend (Unity)
    and simply holds an int reference to tensor in Unity memory space, and a backend controller.
    """

    def __init__(self, controller, data, autograd=False, data_is_pointer=False, verbose=False):
        self.controller = controller
        self.verbose = verbose
        self.id = None

        if data is None:
            raise ValueError("None data provided to FloatTensor.__init__()")

        if data_is_pointer:
            self.id = int(data)
        else:
            if isinstance(data, list):
                data = np.array(data)
            data = data.astype('float')

            controller.socket.send_json({"objectType": "tensor",
                                         "functionCall": "create",
                                         "data": list(data.flatten()),
                                         "shape": data.shape})
            res = json.loads(controller.socket.recv_string())
            # self.id = int(controller.socket.recv_string())
            if not res['ok']:
                raise RuntimeError("Could not create Tensor: {}".format(res['error']))
            if res.get('objectType') != 'tensorID':
                raise RuntimeError("Unexpected message from the backend: {}".format(res))
            self.id = res['id']
            if verbose:
                print("FloatTensor.__init__: " +  str(self.id))

        if autograd:
            self.autograd(True)



    #
    # standard management
    #
    def __del__(self):
        self.delete_tensor()

    def __repr__(self):
        temp = self.to_numpy()
        tensor_str = str(temp)
        type_str = "x".join([str(dim) for dim in temp.shape])
        return tensor_str + "\n<syft.FloatTensor #{}, size {}>".format(self.id, type_str)

    def __str__(self):
        temp = self.to_numpy()
        tensor_str = str(temp).replace('\n', ',')
        type_str = "x".join([str(dim) for dim in temp.shape])
        return tensor_str + "\nsyft.FloatTensor #{}, size {}".format(self.id, type_str)
        # return self.no_params_func("print", True, False)


    #
    # management, state, and repr
    #
    def cpu(self):
        self._manage("cpu")
        return self
        # return self.no_params_func("cpu")

    def copy(self):
        return self._op("copy")
        # return self.no_params_func("copy", return_response=True)

    def delete_tensor(self):
        # this is kept separate as per https://github.com/OpenMined/PySyft/issues/402
        if self.id:
            self._manage("delete")
            # self.no_params_func("delete")

    def gpu(self):
        self._manage("gpu")
        return self
        # return self.no_params_func("gpu")



    #
    # properties get(...), set(...): no compute cost
    #
    def dataOnGpu(self):
        return self._get_prop("dataOnGpu")  # TODO: rename as is_on_gpu
        # return self.get("dataOnGpu") == "1"

    def is_contiguous(self):
        return self._get_prop("is_contiguous")
        # return self.no_params_func("is_contiguous", return_response=True, return_as_tensor=False)

    def size(self):
        """Number of elements

        Returns:
            int: number of tensor elements
        """
        return self._get_prop("size")
        # return int(self.get("size"))

    def shape(self):
        """Returns the size of the self tensor as a np.array."""
        return self._get_prop("shape")
        # shape_tensor = self.no_params_func("size", return_response=True)
        # if as_list:
        #     return [int(x) for x in shape_tensor.get("data").split(",")[:-1]]
        #     # return list(map(lambda x:int(x), shape_tensor.get("data").split(",")[:-1]))
        # return shape_tensor

    def to_numpy(self):
        """Returns the self tensor as a np.array."""
        return self._get_prop("to_numpy")

        # self.controller.socket.send_json({
        #     'functionCall': 'to_numpy',
        #     'objectType': 'tensor',
        #     'objectIndex': self.id
        # })

        # res = self.controller.socket.recv_string()
        # return np.fromstring(res, sep=' ').astype('float').reshape(self.shape())

    # def to_numpy(self):
    #     return self.no_params_func("to_numpy", return_response=True, return_as_tensor=False)



    #
    # dimension manipulations
    #
    def squeeze(self, dim=-1):
        return self.params_func("squeeze", [dim], return_response=True)

    def squeeze_(self, dim=-1):
        return self.params_func("squeeze_", [dim])

    def view(self, *args):
        new_dims = list(args)
        # assert isinstance(new_dim, list)
        assert all([isinstance(dim, int) for dim in new_dims])
        return self.params_func("view", new_dims, return_response=True)

    def view_(self, *args):
        new_dims = list(args)
        # assert isinstance(new_dim, list)
        assert all([isinstance(dim, int) for dim in new_dims])
        self.params_func("view_", new_dims, return_response=False)
        return self

    def view_as(self, x):
        assert isinstance(x, FloatTensor)
        return self.params_func("view_as", [x.id], return_response=True)

    def view_as_(self, x):
        assert isinstance(x, FloatTensor)
        self.params_func("view_as_", [x.id], return_response=False)
        return self

    def T(self):
        return self.no_params_func("transpose", return_response=True)



    #
    # reductions
    #
    def max(self, dim=-1, keepdim=False):
        return self.params_func("max", [dim, keepdim], return_response=True)

    def mean(self, dim=-1, keepdim=False):
        return self.params_func("mean", [dim, keepdim], return_response=True)

    def min(self, dim=-1, keepdim=False):
        return self.params_func("min", [dim, keepdim], return_response=True)

    def prod(self, dim=-1, keepdim=False):
        return self.params_func("prod", [dim, keepdim], return_response=True)

    def sum(self, dim=-1, keepdim=False):
        return self.params_func("sum", [dim, keepdim], return_response=True)

    def sum(self, dim):
        assert isinstance(dim, int)
        return self.arithmetic_operation(dim, "sum", False)

    def trace(self):
        return self._op("trace")
        # return self.no_params_func("trace", return_response=True)




    #
    # unary element-wise operations
    #
    def abs(self):
        return self._op("abs")
        # return self.no_params_func("abs", return_response=True)

    def abs_(self):
        self._op_inline("abs_")
        return self
        # return self.no_params_func("abs_")

    def acos(self):
        return self._op("acos")
        # return self.no_params_func("acos", return_response=True)

    def acos_(self):
        self._op_inline("acos_")
        return self
        # return self.no_params_func("acos_")

    def asin(self):
        return self._op("asin")
        # return self.no_params_func("asin", return_response=True)

    def asin_(self):
        self._op_inline("asin_")
        return self
        # return self.no_params_func("asin_")

    def atan(self):
        return self._op("atan")
        # return self.no_params_func("atan", return_response=True)

    def atan_(self):
        self._op_inline("atan_")
        return self
        # return self.no_params_func("atan_")

    def ceil(self):
        return self._op("ceil")
        # return self.no_params_func("ceil", return_response=True)

    def ceil_(self):
        self._op_inline("ceil_")
        return self
        # return self.no_params_func("ceil_")

    def cos(self):
        return self._op("cos")
        # return self.no_params_func("cos", return_response=True)

    def cos_(self):
        self._op_inline("cos_")
        return self
        # return self.no_params_func("cos_")

    def cosh(self):
        return self._op("cosh")
        # return self.no_params_func("cosh", return_response=True)

    def cosh_(self):
        self._op_inline("cosh_")
        return self
        # return self.no_params_func("cosh_")

    def exp(self):
        return self._op("exp")
        # return self.no_params_func("exp", return_response=True)

    def exp_(self):
        self._op_inline("exp_")
        return self
        # return self.no_params_func("exp_")

    def frac(self):
        return self._op("frac")
        # return self.no_params_func("frac", return_response=True)

    def frac_(self):
        self._op_inline("frac_")
        return self
        # return self.no_params_func("frac_")

    def floor(self):
        return self._op("floor")
        # return self.no_params_func("floor", return_response=True)

    def floor_(self):
        self._op_inline("floor_")
        return self
        # return self.no_params_func("floor_")

    def log(self):
        return self._op("log")
        # return self.no_params_func("log", return_response=True)

    def log_(self):
        self._op_inline("log_")
        return self
        # return self.no_params_func("log_")

    def log1p_(self):
        # return self.no_params_func("log1p_")
        self._op_inline("log1p_")
        return self

    def log1p(self):
        # return self.no_params_func("log1p", return_response=True)
        return self._op("log1p")

    def neg(self):
        return self._op("neg")
        # return self.no_params_func("neg", return_response=True)

    def neg_(self):
        # return self.no_params_func("neg_")
        self._op_inline("neg_")
        return self

    def reciprocal(self):
        return self._op("reciprocal")
        # return self.no_params_func("reciprocal", return_response=True)

    def reciprocal_(self):
        self._op_inline("reciprocal_")
        return self
        # return self.no_params_func("reciprocal_")

    def round(self):
        return self._op("round")
        # return self.no_params_func("round", return_response=True)

    def round_(self):
        # return self.no_params_func("round_")
        self._op_inline("round_")
        return self

    def rsqrt(self):
        return self._op("rsqrt")
        # return self.no_params_func("rsqrt", return_response=True)

    def rsqrt_(self):
        self._op_inline("rsqrt_")
        return self
        # return self.no_params_func("rsqrt_")

    def sigmoid(self):
        return self._op("sigmoid")
        # return self.no_params_func("sigmoid", return_response=True)

    def sigmoid_(self):
        self._op_inline("sigmoid_")
        return self
        # return self.no_params_func("sigmoid_")

    def sign(self):
        return self._op("sign")
        # return self.no_params_func("sign", return_response=True)

    def sign_(self):
        self._op_inline("sign_")
        return self
        # return self.no_params_func("sign_")

    def sin(self):
        return self._op("sin")
        # return self.no_params_func("sin", return_response=True)

    def sin_(self):
        self._op_inline("sin_")
        return self
        # return self.no_params_func("sin_")

    def sinh(self):
        return self._op("sinh")
        # return self.no_params_func("sinh", return_response=True)

    def sinh_(self):
        self._op_inline("sinh_")
        return self
        # return self.no_params_func("sinh_")

    def stride(self, dim=-1):
        if dim == -1:
            return self.no_params_func("stride", return_response=True, return_as_tensor=False)
        else:
            strides = self.params_func("stride", [dim], return_response=True, return_as_tensor=False)
            return np.fromstring(strides, sep=' ').astype('long')

    def sqrt(self):
        return self._op("sqrt")
        # return self.no_params_func("sqrt", return_response=True)

    def trunc(self):
        return self._op("trunc")
        # return self.no_params_func("trunc", return_response=True)

    def tan(self):
        return self._op("tan")
        # return self.no_params_func("tan", return_response=True)

    def tan_(self):
        self._op_inline("tan_")
        return self
        # return self.no_params_func("tan_")

    def tanh(self):
        return self._op("tanh")
        # return self.no_params_func("tanh", return_response=True)

    def triu(self, k=0):
        return self._op("triu", k)
        # return self.params_func("triu", [k], return_response=True)

    def triu_(self, k=0):
        self._op_inline("triu_", k)
        return self
        # return self.params_func("triu_", [k])

    def zero_(self):
        """Fills this tensor with zeros."""
        self._op_inline("zero_")
        return self
        # return self.no_params_func("zero_")



    #
    # n-ary arithmetic operations
    #

    # def arithmetic_operation(self, x, name, inline=False):
    #     operation_cmd = name
    #
    #     if isinstance(x, FloatTensor):
    #         operation_cmd += "_elem"
    #         parameter = x.id
    #     else:
    #         operation_cmd += "_scalar"
    #         parameter = str(x)
    #
    #     if inline:
    #         operation_cmd += "_"
    #
    #     self.controller.socket.send_json(
    #         self.cmd(operation_cmd, [parameter]))  # sends the command
    #     return FloatTensor(controller=self.controller, data=int(self.controller.socket.recv_string()), data_is_pointer=True)

    def __add__(self, x):
        name, param = self._build_arithmetic_op(x, "add")
        return self._op(name, param)
        # return self.arithmetic_operation(x, "add", False)

    def __iadd__(self, x):
        name, param = self._build_arithmetic_op(x, "add_")
        self._op_inline(name, param)
        return self
        # return self.arithmetic_operation(x, "add", True)

    def addmm_(self, x, y):
        self._op_inline("addmm_", [x.id, y.id])
        return self
        # return self.params_func("addmm_", [x.id, y.id])

    def addmm(self, x, y):
        copy = self.copy()
        copy.addmm_(x, y)
        # copy.params_func("addmm_", [x.id, y.id])
        return copy

    def addmv_(self, x, y):
        self._op_inline("addmv_", [x.id, y.id])
        return self
        # return self.params_func("addmv_", [x.id, y.id])

    def addmv(self, x, y):
        copy = self.copy()
        copy.addmv_(x, y)
        # copy.params_func("addmv_", [x.id, y.id])
        return copy

    def __truediv__(self, x):
        name, param = self._build_arithmetic_op(x, "div")
        return self._op(name, param)
        # return self.arithmetic_operation(x, "div", False)

    def __itruediv__(self, x):
        name, param = self._build_arithmetic_op(x, "div_")
        self._op_inline(name, param)
        return self
        # return self.arithmetic_operation(x, "div", True)

    def __pow__(self, x):
        name, param = self._build_arithmetic_op(x, "pow")
        return self._op(name, param)
        # return self.arithmetic_operation(x, "pow", False)

    def __ipow__(self, x):
        name, param = self._build_arithmetic_op(x, "pow_")
        self._op_inline(name, param)
        return self
        # return self.arithmetic_operation(x, "pow", True)

    def __mod__(self, divisor):
        return self.remainder(divisor)
        # return self.arithmetic_operation(x, "remainder", False)

    def __imod__(self, divisor):
        self.remainder_(divisor)
        return self

    def __mul__(self, x):
        name, param = self._build_arithmetic_op(x, "mul")
        return self._op(name, param)
        # return self.arithmetic_operation(x, "mul", False)

    def __imul__(self, x):
        name, param = self._build_arithmetic_op(x, "mul_")
        self._op_inline(name, param)
        return self
        # return self.arithmetic_operation(x, "mul", True)

    def mm(self, x):
        return self._op("mm", x.id)
        # return self.params_func("mm", [x.id], True)

    def pow(self, x):
        name, param = self._build_arithmetic_op(x, "pow")
        return self._op(name, param)
        # return self.arithmetic_operation(x, "pow", False)

    def pow_(self, x):
        name, param = self._build_arithmetic_op(x, "pow_")
        self._op_inline(name, param)
        return self
        # return self.arithmetic_operation(x, "pow", True)

    def remainder(self, divisor):
        name, param = self._build_arithmetic_op(divisor, "remainder")
        return self._op(name, param)
        # return self.arithmetic_operation(divisor, "remainder", False)

    def remainder_(self, divisor):
        name, param = self._build_arithmetic_op(divisor, "remainder_")
        self._op_inline(name, param)
        return self
        # return self.arithmetic_operation(divisor, "remainder", True)

    def __sub__(self, x):
        name, param = self._build_arithmetic_op(x, "sub")
        return self._op(name, param)
        # return self.arithmetic_operation(x, "sub", False)

    def __isub__(self, x):
        name, param = self._build_arithmetic_op(x, "sub_")
        self._op_inline(name, param)
        return self
        # return self.arithmetic_operation(x, "sub", True)



    #
    # autograd
    #
    def autograd(self, setter=None):
        if setter is None:
            return self._get_prop("autograd")

        assert isinstance(setter, bool)
        self._set_prop("autograd", setter)  # no check: an exception is raised if this op failed
        return self

        # if setter is None:
        #     if self.get("autograd") == "1":
        #         return True
        #     return False
        # else:
        #     if setter:
        #         out = self._set_prop("autograd", True)
        #         # out = self.set("autograd", ["1"])
        #     else:
        #         out = self._set_prop("autograd", False)
        #         # out = self.set("autograd", ["0"])

        #     if (out == "1" and setter) or (out == "0" and not setter):
        #         return self
        #     return False

    def backward(self, grad=None):
        # pure side-effect
        if grad is None:
            self._op_inline("backward")
            # self.no_params_func("backward")
        else:
            self._op_inline("backward", grad.id)
            # self.params_func(name="backward", params=[grad.id])

    def children(self):
        return self._get_prop("children")  # list of tensor ids
        # res = self.get("children")
        # if len(res) > 0:
        #     return list(map(lambda x:int(x), res.split(",")[:-1]))
        # return []

    def creation_op(self):
        return self._get_prop("creation_op")  # string
        # return self.get("creation_op")

    def creators(self):
        return self._get_prop("creators")  # list of tensor ids
        # res = self.get("creators")
        # if len(res) > 0:
        #     return list(map(lambda x:int(x), res.split(",")[:-1]))
        # return []

    def keepgrad(self):
        return self._get_prop("keepgrads")  # bool
        # return self.get("keepgrad") == "1"

    def grad(self):
        return self._get_prop("grad")  # tensor
        # return self.get("grad", response_as_tensor=True)



    """
    Helper methods that
    - sets semantic operations: _op, _op_inline, _get_prop, _set_prop, _manage
    - sets clearer return types: avoid leaking comm details (strings) up to high-level code.

    To achieve this, it handles inbound and outbound messaging with the controller.
    """
    def _cmd(self, functionCall, tensorIndexParams=None):
        return {
            'functionCall': functionCall,
            'objectType': 'tensor',
            'objectIndex': self.id,
            'tensorIndexParams': tensorIndexParams or []}


    def _remote_execute(self, name, params):
        """
        Synchronous execution, may be extended with timeout for instance.
        This basically replaces params_func.
        """
        # send the command
        self.controller.socket.send_json(
            self._cmd(name, tensorIndexParams=params))

        # receive output from command
        raw_response = self.controller.socket.recv_string()

        if self.verbose:
            print("command: " + name + " -> " + raw_response)

        return raw_response


    def _parse(self, raw_response, typeassert=None):
        """
        Parse controller response string -> json -> types
        """
        response = json.loads(
            raw_response.replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t'))

        if not response['ok']:
            return None, response.get('error')

        objectType = response['objectType']
        #, response['objectParams']

        if objectType == 'status':
            result = response['ok']  # which is true

        elif objectType == 'numpy':
            # objectShape = [int(dim) for dim in objectParams.split(',')]
            shape = response['shape']
            result = np.array(response['data']).astype('float').reshape(shape)
            # result = np.fromstring(strValue, sep=',').astype('float').reshape(objectShape)

        elif objectType in ['string', 'int', 'float', 'bool', 'intList']:
            result = response['value']

        elif objectType == 'tensorID':
            tid = response['id']
            result = FloatTensor(controller=self.controller, data=tid, data_is_pointer=True)
            # declare t with controller?

        elif objectType == 'tensorIDList':
            result = response['ids']

        if typeassert:
            assert isinstance(result, tuple(typeassert))

        return result, None


    def _op(self, name, params=None):
        """
        Returns a FloatTensor, a float or a list as a result of pure function application (no side-effect).
        """
        if params and not isinstance(params, list):
            params = [params]
        retval, error = self._parse(self._remote_execute(name, params), typeassert=[FloatTensor, float])
        if error:
            raise RuntimeError("Command {} failed: {}".format(name, error))
        return retval

    def _op_inline(self, name, params=None):
        """
        Inline variant of `op`: executes operation `name` and returns nothing.

        An exception is raised in case the operation failed remotely.
        """
        if params and not isinstance(params, list):
            params = [params]
        # if not name.endswith("_"):
        #     name += "_"
        ok, error = self._parse(self._remote_execute(name, params), typeassert=[bool])
        if error or not ok:
            raise RuntimeError("Command {} failed: {}".format(name, error))

    def _get_prop(self, param_name="size"):
        """
        Basically reads an existing property on Unity side of the bridge. The difference with op is
        that no computation is required: the property already exists as such on Unity side.
        The returned property could be any type (list, float, bool), so no type assertions are made here:
        get_prop can almost return anything
        """
        retval, error = self._parse(self._remote_execute("get", [param_name]), typeassert=None)
        if error:
            raise RuntimeError("Command get_prop with param {} failed: {}".format([param_name], error))
        return retval

    def _set_prop(self, param_name="size", params=None):
        """
        Basically writes a property on Unity side. The semantics is that no computation is required.
        Returns nothing, but raises in case of failure on Unity side.
        """
        if params and not isinstance(params, list):
            params = [params]
        prm = [param_name] + params
        ok, error = self._parse(self._remote_execute("set", prm), typeassert=[bool])
        if error or not ok:
            raise RuntimeError("Command set_prop with param {} failed: {}".format(prm, error))

    def _manage(self, name):
        """
        Semantically, this method is for memory management methods: cpu, gpu, delete, ...

        Returns nothing, but raises in case of failure on Unity side.
        """
        ok, error = self._parse(self._remote_execute(name, []), typeassert=[bool])
        if error or not ok:
            raise RuntimeError("Command {} failed: {}".format(name, error))


    def _build_arithmetic_op(self, x, name):
        # NOTE: could be a pure function because self is unused
        inline = name.endswith("_")
        if inline:
            name = name[:-1]

        operation_cmd = name

        if isinstance(x, FloatTensor):
            operation_cmd += "_elem"
            parameter = x.id
        else:
            operation_cmd += "_scalar"
            parameter = x

        if inline:
            operation_cmd += "_"

        return operation_cmd, parameter




class SyftController():
    def __init__(self, verbose=True):

        self.identity = str(uuid.uuid4())

        context = zmq.Context()
        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.identity)
        self.socket.connect("tcp://localhost:5555")
        self.verbose = verbose

    def FloatTensor(self, data, autograd=False):
        return FloatTensor(controller=self, data=data, autograd=autograd, verbose=self.verbose)

    def Linear(self, *args):
        return Linear(sc=self, dims=args)

    def Sigmoid(self):
        return Sigmoid(sc=self)

    def Sequential(self):
        return Sequential(sc=self)

    def rand(self, *args):
        return self.FloatTensor(np.random.rand(*args))

    def randn(self, *args):
        return self.FloatTensor(np.random.randn(*args))

    def zeros(self, *args):
        return self.FloatTensor(np.zeros((args)))

    def ones(self, *args):
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

