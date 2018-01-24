import numpy as np
import syft.controller
from base_tensor import BaseTensor

class IntTensor(BaseTensor):
    def __init__(self, data, data_is_pointer=False):
        self.controller = syft.controller

        if (data is not None and not data_is_pointer):

            if (type(data) == list):
                data = np.array(data)
            data = data.astype('float')

            self.data = data
            self.id = int(self.controller.send_json({"objectType": "IntTensor",
                                                     "functionCall": "create",
                                                     "data": list(data.flatten()),
                                                     "shape": self.data.shape}))
        elif (data_is_pointer):
            self.id = int(data)

    def autograd(self, state):
        "do nothing"

    def abs(self):
        """
        Returns absolute value of tensor as a new tensor
        Parameters
        ----------
        Returns
        -------
        IntTensor:
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
        IntTensor
            Output tensor
        """
        return self.no_params_func("abs_", return_response=True)

    def cos(self):
        """
        Computes cos of each element of the tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output Tensor
        """
        return self.no_params_func("cos", return_response=True, return_type='FloatTensor')

    def lt(self, other):
        """
        Performs element-wise > comparison and returns 1 if the element
        is less than a corresponding element in other Tensor, and 0 otherwise.
        Returns a new Tensor with results of the comparison.

        Parameters
        __________
        other: IntTensor to compare with

        Returns
        _________
        IntTensor
            Output tensor
        """
        return self.params_func("lt", [other.id], return_response=True)

    def lt_(self, other):
        """
        Performs inline element-wise > comparison and returns 1 if the element
        is less than a corresponding element in other Tensor, and 0 otherwise.

        Parameters
        __________
        other: IntTensor to compare with

        Returns
        _________
        IntTensor
            Output tensor
        """
        return self.params_func("lt_", [other.id], return_response=True)

    def eq(self, other):
        """
        Determines whether 'other' (IntTensor) has the same elements as self (IntTensor).

        parameters:
            other: IntTensor, of the same dimension as self
        returns: IntTensor, with values
            1 - when the elements are equal
            0 - when the elements are not equal
        """
        return self.params_func("eq", [other.id], return_response=True)

    def eq_(self, other):
        """
        Determines whether 'other' (IntTensor) has the same elements as self (IntTensor).

        parameters:
            other: IntTensor, of the same dimension as self
        returns: IntTensor, with values
            1 - when the elements are equal
            0 - when the elements are not equal
        """
        return self.params_func("eq_", [other.id], return_response=True)

    def equal(self, x):
        """
        Determines whether the given tensor has the same size and elements as this instance.

        :param x: IntTensor
        :return: True if the given tensor has the same size and elements as this instance. Otherwise, False.
        """
        response_string = self.params_func("equal", [x.id], return_response=True, return_type="str")
        if response_string == "True":
            return True
        else:
            return False

    def neg(self):
        """
        Sets negative of the elements of tensor.
        Parameters
        ----------
        Returns
        -------
        IntTensor
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
        IntTensor
            Caller with values inplace
        """
        return self.no_params_func("neg_")

    def shape(self):
        """
        Returns the size of the self tensor as a List.

        Returns
        -------
        Iterable
            Output list
        """
        return list(np.fromstring(self.get("shape")[:-1], sep=",").astype('int'))

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

    def reciprocal(self):
        """
        Computes the reciprocal of the input tensor.
        ----------
        Returns
        -------
        IntTensor:
            Output Tensor
        """
        return self.no_params_func("reciprocal", return_response=True)

    def reciprocal_(self):
        """
        Computes reciprocal of input tensor with values inplace.
        Parameters
        ----------
        Returns
        -------
        IntTensor
            Caller with values inplace
        """
        return self.no_params_func("reciprocal_")

    def trace(self):
        """
        Returns a new tensor with the sum along diagonals of a 2D tensor.
        Returns
        -------
        IntTensor
            Output tensor
        """
        return self.no_params_func("trace", return_response=True)

    def sin(self):
        """
        Computes sin of each element of the tensor.
        Parameters
        ----------
        Returns
        -------
        FloatTensor
            Output Tensor
        """
        return self.no_params_func("sin", return_response=True, return_type='FloatTensor');

    def __repr__(self, verbose=True):

        tensor_str = str(self.to_numpy())

        type_str = ""
        for dim in self.shape():
            type_str += str(dim) + "x"

        type_str = type_str[:-1]

        desc = "[syft.IntTensor:"+str(self.id) + " size:" + type_str + "]" + "\n"

        return tensor_str + "\n" + desc

    def T(self, dim1=None, dim2=None):
        """
        Returns a tensor that is a transposed version of input. The given dimensions dim1 and dim0 are swapped.
        Parameters:

            input (Tensor) – the input IntTensor
            dim0 (int) – the first dimension to be transposed
            dim1 (int) – the second dimension to be transposed
        ----------
        Returns
        -------
        IntTensor
            Output tensor
        """
        if isinstance(dim1, int) and isinstance(dim2, int):
            return self.params_func("transpose", [dim1, dim2], return_response=True)
        else:
            return self.no_params_func("transpose", return_response=True)

    def params_func(self, name, params, return_response=False, return_type='IntTensor'):
        # send the command
        res = self.controller.send_json(
            self.cmd(name, params=params))

        self.controller.log(res)

        if (return_response):
            if (return_type == 'IntTensor'):
                self.controller.log("IntTensor.__init__: {}".format(res))
                return IntTensor(data=int(res), data_is_pointer=True)
            elif(return_type == 'FloatTensor'):
                self.controller.log("IntTensor.__init__: {}".format(res))
                return FloatTensor(data=int(res), data_is_pointer=True)
            else:
                return res
        return self

    def no_params_func(self, name, return_response=False, return_type='IntTensor'):
        return (self.params_func(name, [], return_response, return_type))

    def get(self, param_name="size", response_as_tensor=False, return_type='IntTensor'):
        return self.params_func(name="get", params=[param_name], return_response=True,
                                return_type="string")

    def cmd(self, functionCall, params=[]):
        cmd = {
            'functionCall': functionCall,
            'objectType': 'IntTensor',
            'objectIndex': self.id,
            'tensorIndexParams': params}
        return cmd

    def gpu(self):
        """
        Returns a GPU copy of this storage if it's not already on the GPU
        Parameters
        ----------
        Returns
        -------
        IntTensor
            Output tensor
        """
        return self.no_params_func("gpu")

    def is_contiguous(self):
        return True

    def to_numpy(self):
        if(self.is_contiguous()):
            res = self.controller.send_json({
                'functionCall': 'to_numpy',
                'objectType': 'IntTensor',
                'objectIndex': self.id
            })

            return np.fromstring(res, sep=' ').astype('int').reshape(self.shape())
        else:
            return " - non-contiguous - "

    def sign(self):
        """
        Computes sign of each element of the tensor.
        Parameters
        ----------
        Returns
        -------
        IntTensor
            Output tensor
        """
        return self.no_params_func("sign", return_response=True)