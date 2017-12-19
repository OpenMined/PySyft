import numpy as np

import syft.controller


class FloatTensor():
    def __init__(self, data, autograd=False, data_is_pointer=False):
        self.controller = syft.controller

        if (data is not None and not data_is_pointer):

            if (type(data) == list):
                data = np.array(data)
            data = data.astype('float')

            self.data = data
            self.id = int(self.controller.send_json({"objectType": "tensor",
                                                     "functionCall": "create",
                                                     "data": list(data.flatten()),
                                                     "shape": self.data.shape}))
            # self.controller.log("FloatTensor.__init__: {}".format(self.id))

        elif (data_is_pointer):
            self.id = int(data)

        if (autograd):
            self.autograd(True)

            # def __del__(self):
            # self.delete_tensor()

    def abs(self):
        """
        Returns absolute value of tensor as a new tensor
        Parameters
        ----------
        Returns
        -------
        FloatTensor:
            Output tensor
        """
        return self.no_params_func("abs", return_response=True)

    def abs_(self):
        """
        Replaces tensor values with its absolute value
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("abs_")

    def acos(self):
        """
        Returns a new Tensor with the arccosine of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("acos", return_response=True)

    def acos_(self):
        """
        Performs inplace arccosine operation of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("acos_")

    def addmm_(self, x, y):
        """
        Performs a matrix multiplication of the matrices 'x' and 'y'.
        The caller matrix 'self' is added to the final result inplace.
        Parameters
        ----------
        x : FloatTensor
            First tensor for multiplication
        y : FloatTensor
            Second tensor for multiplication
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.params_func("addmm_", [x.id, y.id])

    def addmm(self, x, y):
        """
        Performs a matrix multiplication of the matrices 'x' and 'y'.
        The caller matrix 'self' is added to the final result.
        Parameters
        ----------
        x : FloatTensor
            First tensor for multiplication
        y : FloatTensor
            Second tensor for multiplication
        Returns
        -------
        copy : FloatTensor
            Output tensor
        """
        copy = self.copy()
        copy.params_func("addmm_", [x.id, y.id])
        return copy

    def addmv_(self, x, y):
        """
        Performs a matrix-vector product of the matrix x and the vector vec.
        The vector tensor is added to the final result inplace.
        Parameters
        ----------
        x : FloatTensor
            tensor for multiplication
        vec : FloatTensor
            Vector for Matrix-Vector Product
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.params_func("addmv_", [x.id, y.id])

    def addmv(self, x, y):
        """
        Performs a matrix-vector product of the matrix x and the vector vec.
        The vector tensor is added to the final result.
        Parameters
        ----------
        x : FloatTensor
            tensor for multiplication
        y : FloatTensor
            Vector for Matrix-Vector Product
        Returns
        -------
        copy : FloatTensor
            Output tensor
        """
        copy = self.copy()
        copy.params_func("addmv_", [x.id, y.id])
        return copy

    def asin(self):
        """
        Returns a new Tensor with the arcsine of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("asin", return_response=True)

    def asin_(self):
        """
        Performs inplace arcsine operation of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("asin_")

    def atan(self):
        """
        Returns a new Tensor with the arctangent of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("atan", return_response=True)

    def atan_(self):
        """
        Performs inplace arctangent operation of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("atan_")

    def autograd(self, setter=None):
        if (setter is None):
            if (self.get("autograd") == "1"):
                return True
            else:
                return False
        else:
            if (setter):
                out = self.set("autograd", ["1"])
            else:
                out = self.set("autograd", ["0"])

            if (out == "1" and setter) or (out == "0" and not setter):
                return self
            else:
                return False

    def __add__(self, x):
        """
        Performs element-wise addition arithmetic between two tensors
        Parameters
        ----------
        x : FloatTensor
            The Second tensor to perform addition with.
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.arithmetic_operation(x, "add", False)

    def __iadd__(self, x):
        """
        Performs in place element-wise addition arithmetic between two tensors
        Parameters
        ----------
        x : FloatTensor
            The Second tensor to perform addition with.
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "add", True)

    def backward(self, grad=None):
        if (grad is None):
            self.no_params_func("backward")
        else:
            self.params_func(name="backward", params=[grad.id])

    def ceil(self):
        """
        Performs the ceiling of the input tensor element-wise.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("ceil", return_response=True)

    def ceil_(self):
        """
        Performs the inplace ceiling of the input tensor element-wise.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("ceil_")

    def copy(self):
        """
        Returns a copy of the input
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("copy", return_response=True)

    def cos(self):
        """
        Returns a new Tensor with the cosine of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("cos", return_response=True)

    def cos_(self):
        """
        Performs the cosine of the input tensor inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("cos_")

    def cosh(self):
        """
        Returns a new Tensor with hyperbolic cosine of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("cosh", return_response=True)

    def cosh_(self):
        """
        Returns the hyperbolic cosine of the input inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("cosh_")

    def children(self):
        """
        Returns an iterator over immediate children modules.
        Parameters
        ----------
        Returns
        -------
        Iterable
            Returns a list of children
        """
        res = self.get("children")
        if (len(res) > 0):
            return list(map(lambda x: int(x), res.split(",")[0:-1]))
        return []

    def creation_op(self):
        return self.get("creation_op")

    def creators(self):
        """
        Returns an iterator over immediate creators of input tensor.
        Parameters
        ----------
        Returns
        -------
        Returns a list of creators
        """
        res = self.get("creators")
        if (len(res) > 0):
            return list(map(lambda x: int(x), res.split(",")[0:-1]))
        return []

    def dataOnGpu(self):
        if (self.get("dataOnGpu") == "1"):
            return True
        return False

    def exp(self):
        """
        Computes the exponential of each element of input tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("exp", return_response=True)

    def exp_(self):
        """
        Computes the exponential of each element of input tensor inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("exp_")

    def __truediv__(self, x):
        """
        Performs division arithmetic between two tensors
        Parameters
        ----------
        x : FloatTensor
            Second divident tensor
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.arithmetic_operation(x, "div", False)

    def __itruediv__(self, x):
        """
        Performs division arithmetic between two tensors inplace.
        Parameters
        ----------
        x : FloatTensor
            Second divident tensor
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "div", True)

    def keepgrad(self):
        if (self.get("keepgrad") == "1"):
            return True
        else:
            return False

    def __pow__(self, x):
        """
        Takes the power of each element in input ('self') with 'x' and
        returns a tensor with the result.
        Parameters
        ----------
        x : FloatTensor
            Exponent tensor
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.arithmetic_operation(x, "pow", False)

    def __ipow__(self, x):
        """
        Takes the power of each element in input ('self') with 'x' and
        returns a tensor with the result inplace.
        Parameters
        ----------
        x : FloatTensor
            Exponent tensor
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "pow", True)

    def pow(self, x):
        """
        Takes the power of each element in input ('self') with 'x' and
        returns a tensor with the result.
        Parameters
        ----------
        x : FloatTensor
            Exponent tensor
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.arithmetic_operation(x, "pow", False)

    def pow_(self, x):
        """
        Takes the power of each element in input ('self') with 'x', inplace.
        Parameters
        ----------
        x : FloatTensor
            Exponent tensor
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "pow", True)

    def floor(self):
        """
        Performs the floor of the input tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("floor", True)

    def floor_(self):
        """
        Performs the inplace floor of the input tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("floor_")

    def round(self):
        """
        Performs Round-ing to the nearest decimal,
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("round", return_response=True)

    def round_(self):
        """
        Performs Round-ing to the nearest decimal inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("round_")

    def mm(self, other):
        """
        Performs a matrix multiplication of two tensors.
        Parameters
        ----------
        other : FloatTensor
            Second tensor to be multiplied with
        Returns
        -------
        FloatTensor
            n x m Output tensor
        """
        return self.params_func("mm", [other.id], True)

    def grad(self):
        return self.get("grad", response_as_tensor=True)

    def __mod__(self, x):
        """
        Performs Modulus arithmetic operation between two tensors.
        Parameters
        ----------
        x : FloatTensor
            Dividend tensor
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.arithmetic_operation(x, "remainder", False)

    def __imod__(self, x):
        """
        Performs Modulus arithmetic operation between two tensors inplace.
        Parameters
        ----------
        x : FloatTensor
            Dividend tensor
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "remainder", True)

    def __mul__(self, x):
        """
        Performs Multiplication arithmetic operation between two tensors.
        Parameters
        ----------
        x : FloatTensor
            Second tensor to be multiplied with.
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.arithmetic_operation(x, "mul", False)

    def __imul__(self, x):
        """
        Performs Multiplication arithmetic operation between two tensors inplace.
        Parameters
        ----------
        x : FloatTensor
            Second tensor to be multiplied with.
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "mul", True)

    def neg(self):
        """
        Sets negative of the elements of tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("neg", return_response=True)

    def neg_(self):
        """
        Sets negative of the elements of tensor inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("neg_")

    def rsqrt(self):
        """
        Returns reciprocal of square root of tensor element wise.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("rsqrt", return_response=True)

    def set(self, param_name="size", params=[]):
        return self.params_func(name="set", params=[param_name] + params, return_response=True, return_as_tensor=False)

    def sigmoid_(self):
        """
        Performs inplace sigmoid function on the tensor element-wise.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace.
        """
        return self.no_params_func("sigmoid_")

    def sigmoid(self):
        """
        Returns a new tensor holding element wise values of Sigmoid function.
        Sigmoid(x) = 1 / 1+exp(-x)
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("sigmoid", return_response=True)

    def sign(self):
        """
        Computes sign of each element of the tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("sign", return_response=True)

    def sign_(self):
        """
        Computes the sign of each element of the tensor inplace
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("sign_")

    def sin(self):
        """
        Computes sin of each element of the tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("sin", return_response=True)

    def sin_(self):
        """
        Computes the sine of each element of the tensor inplace
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("sin_")

    def size(self):
        """
        Returns the size of tensor.
        Parameters
        ----------
        Returns
        -------
        int
            int with value of size
        """
        return int(self.get("size"))

    def shape(self, as_list=True):
        """
        Returns the size of the self tensor as a FloatTensor (or as List).
        Note:
            The returned value currently is a FloatTensor because it leverages
            the messaging mechanism with Unity.
        Parameters
        ----------
        as_list : bool
            Value retruned as list if true; else as tensor
        Returns
        -------
        FloatTensor
            Output tensor
        (or)
        Iterable
            Output list
        """
        if (as_list):
            return list(np.fromstring(self.get("shape")[:-1], sep=",").astype('int'))
        else:
            shape_tensor = self.no_params_func("shape", return_response=True)
            return shape_tensor

    def stride(self, dim=-1):
        """
        Returns the stride of tensor.
        Parameters
        ----------
        dim : int
            dimension of expected return

        Returns
        -------
        FloatTensor
            Output tensor.
        (or)
        numpy.ndarray
            NumPy Array as Long
        """
        if dim == -1:
            return self.no_params_func("stride", return_response=True, return_as_tensor=False)
        else:
            strides = self.params_func("stride", [dim], return_response=True, return_as_tensor=False)
            return np.fromstring(strides, sep=' ').astype('long')

    def sqrt(self):
        """
        Returns a new tensor with the square-root of the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor:
            Output Tensor
        """
        return self.no_params_func("sqrt", return_response=True)

    def trace(self):
        """
        Returns a new tensor with the sum along diagonals of a 2D tensor.
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("trace", return_response=True)

    def trunc(self):
        return self.no_params_func("trunc", return_response=True)

    def to_numpy(self):
        res = self.controller.send_json({
            'functionCall': 'to_numpy',
            'objectType': 'tensor',
            'objectIndex': self.id
        })

        return np.fromstring(res, sep=' ').astype('float').reshape(self.shape())

    def __sub__(self, x):
        """
        Performs element-wise substraction arithmetic between two tensors
        Parameters
        ----------
        x : FloatTensor
            The Second tensor to perform addition with.
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.arithmetic_operation(x, "sub", False)

    def __isub__(self, x):
        """
        Performs element-wise substraction arithmetic between two tensors
        Parameters
        ----------
        x : FloatTensor
            The Second tensor to perform addition with.
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.arithmetic_operation(x, "sub", True)

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
        """
        Returns a tensor that is a transposed version of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("transpose", return_response=True)

    def triu(self, k=0):
        return self.params_func("triu", [k], return_response=True)

    def triu_(self, k=0):
        return self.params_func("triu_", [k])

    def zero_(self):
        """
        Fills this tensor with zeros inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("zero_")

    def __repr__(self, verbose=True):

        tensor_str = str(self.to_numpy())

        type_str = ""
        for dim in self.shape():
            type_str += str(dim) + "x"

        type_str = type_str[:-1]
        grad = self.get("grad")
        if (grad == ''):
            grad = 'None'
        desc = "[syft.FloatTensor:" + str(self.id) + " grad:" + grad + " size:" + type_str + "]" + "\n"

        if (verbose):
            children = self.children()
            creators = self.creators()
            if (len(children) > 0):
                # tensor_str = "\n -------------------------------\n" + tensor_str
                desc += "\n\t-----------children-----------\n\t"
            for child_id in children:
                desc += syft.controller.get_tensor(child_id).__repr__(False)
            if (len(children) > 0):
                if (len(creators) > 0):
                    desc += "\t------------------------------\n"
                else:
                    desc += "\t------------------------------\n\n\n"

            if (len(creators) > 0):
                # tensor_str = "\n -------------------------------\n" + tensor_str
                desc += "\n\t-----------creators-----------\n"
            for parent_id in creators:
                desc += "\t" + syft.controller.get_tensor(parent_id).__repr__(False)
            if (len(creators) > 0):
                desc += "\t------------------------------\n\n\n"

            return tensor_str + "\n" + desc
        return desc

    def __str__(self):
        tensor_str = str(self.to_numpy()).replace("]", " ").replace("[", " ") + "\n"
        return tensor_str

    def get(self, param_name="size", response_as_tensor=False):
        return self.params_func(name="get", params=[param_name], return_response=True,
                                return_as_tensor=response_as_tensor)

    def cpu(self):
        """
        Returns a CPU copy of this storage if it's not already on the CPU
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("cpu")

    def gpu(self):
        """
        Returns a GPU copy of this storage if it's not already on the GPU
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
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
        res = self.controller.send_json(
            self.cmd(name, tensorIndexParams=params))

        self.controller.log(res)

        if (return_response):
            if (return_as_tensor):
                self.controller.log("FloatTensor.__init__: {}".format(res))
                return FloatTensor(data=int(res), data_is_pointer=True)
            else:
                return res
        return self

    def no_params_func(self, name, return_response=False, return_as_tensor=True):
        return (self.params_func(name, [], return_response, return_as_tensor))

    def arithmetic_operation(self, x, name, inline=False):

        operation_cmd = name

        if (type(x) == FloatTensor):
            operation_cmd += "_elem"
            parameter = x.id
        else:
            operation_cmd += "_scalar"
            parameter = str(x)

        if (inline):
            operation_cmd += "_"

        response = self.controller.send_json(
            self.cmd(operation_cmd, [parameter]))  # sends the command
        return FloatTensor(data=int(response), data_is_pointer=True)

    def delete_tensor(self):
        """
        Deletes the input tensor.
        Parameters
        ----------
        Returns
        -------
        """
        if (self.id is not None):
            self.no_params_func("delete")
        self.controller = None
        self.id = None

    def T(self):
        """
        (Function DUPLICATED?)
        """
        return self.no_params_func("transpose", return_response=True)

    def is_contiguous(self):
        return self.no_params_func("is_contiguous", return_response=True, return_as_tensor=False)

    def sinh(self):
        """
        Returns the hyperbolic sine of the input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("sinh", return_response=True)

    def sinh_(self):
        """
        Returns the hyperbolic sine of the input inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace.
        """
        return self.no_params_func("sinh_")

    def log(self):
        """
        Returns the logarithm of the input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("log", return_response=True)

    def log_(self):
        """
        Returns the logarithm of the input inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace.
        """
        return self.no_params_func("log_")

    def log1p_(self):
        """
        Returns the natural logarithm of (1 + input) inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace.
        """
        return self.no_params_func("log1p_")

    def log1p(self):
        """
        Returns a new tensor with the natural logarithm of (1 + 'self').
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("log1p", return_response=True)

    def frac(self):
        """
        Computes the fractional portion of each element in tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("frac", return_response=True)

    def frac_(self):
        """
        Computes the fractional portion of each element in tensor, inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("frac_")

    def reciprocal(self):
        """
        Computes the reciprocal of the input tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("reciprocal", return_response=True)

    def reciprocal_(self):
        """
        Computes reciprocal of input tensor with values inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("reciprocal_")

    def rsqrt(self):
        """
        Returns a new tensor with the reciprocal of the square-root of each of
        the elements of input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("rsqrt", return_response=True)

    def rsqrt_(self):
        """
        Computes the reciprocal of the square-root of each of the elements of input,
        inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("rsqrt_")

    def remainder(self, divisor):
        """
        Computes the element-wise remainder of division.
        inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.arithmetic_operation(divisor, "remainder")

    def remainder_(self, divisor):
        """
        Computes the element-wise remainder of division, inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.arithmetic_operation(divisor, "remainder", True)

    def tan(self):
        """
        Returns the tangent of the input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("tan", return_response=True)

    def tan_(self):
        """
        Returns the tangent of the input inplace.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.no_params_func("tan_")

    def tanh(self):
        """
        Returns the hyperbolic tangent of the input.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.no_params_func("tanh", return_response=True)

    def squeeze(self, dim=-1):
        """
        Returns a tensor with all the dimensions of input of size 1 removed.
        Parameters
        ----------
        dim : int
            When dim is given, a squeeze operation is done only in the given
            dimension.
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.params_func("squeeze", [dim], return_response=True)

    def squeeze_(self, dim=-1):
        """
        Removes all the dimensions of input tensor of size 1, inplace.
        Parameters
        ----------
        dim : int
            When dim is given, a squeeze operation is done only in the given
            dimension.
        Returns
        -------
        FloatTensor
            Caller with values inplace
        """
        return self.params_func("squeeze_", [dim])

    def min(self, dim=-1, keepdim=False):
        """
        Returns the minimum value of all elements in the input tensor.
        Parameters
        ----------
        dim : int
            the dimension to reduce
        keepdim : bool
            whether the output tensors have dim retained or not
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.params_func("min", [dim, keepdim], return_response=True)

    def max(self, dim=-1, keepdim=False):
        """
        Returns the maximum value of all elements in the input tensor.
        Parameters
        ----------
        dim : int
            the dimension to reduce
        keepdim : bool
            whether the output tensors have dim retained or not
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.params_func("max", [dim, keepdim], return_response=True)

    def sum(self, dim=-1, keepdim=False):
        """
        Returns the sum of all elements in the input tensor.
        Parameters
        ----------
        dim : int
            the dimension to reduce
        keepdim : bool
            whether the output tensors have dim retained or not
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.params_func("sum", [dim, keepdim], return_response=True)

    def prod(self, dim=-1, keepdim=False):
        """
        Returns the product of all elements in the input tensor.
        Parameters
        ----------
        dim : int
            the dimension to reduce
        keepdim : bool
            whether the output tensors have dim retained or not
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.params_func("prod", [dim, keepdim], return_response=True)

    def mean(self, dim=-1, keepdim=False):
        """
        Returns the mean value of all elements in the input tensor.
        Parameters
        ----------
        dim : int
            the dimension to reduce
        keepdim : bool
            whether the output tensors have dim retained or not
        Returns
        -------
        FloatTensor
            Output tensor
        """
        return self.params_func("mean", [dim, keepdim], return_response=True)
