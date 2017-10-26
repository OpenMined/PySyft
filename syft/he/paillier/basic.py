import numpy as np
from ...tensor import TensorBase


class PaillierTensor(TensorBase):

    def __init__(self, public_key, data=None, input_is_decrypted=True, fixed_point_conf=None):
        """
        Initializes a Tensor. If `data` is encrypted, `fixed_point_conf` needs to match that of `data`.
        That is `fixed_point_conf` should be the same one that was used for encryption of `data`.
        If no `fixed_point_conf` is provided, it creates its own default fixed-point configuration.
        :param public_key: Public key to encrypt the data by
        :param data: Array-like data
        :param input_is_decrypted: To indicate whether `data` is encrypted
        :param fixed_point_conf: Precision configuration of type `FXfamily`
        """
        self.encrypted = True
        if fixed_point_conf is None:
            # creates default fixed point configuration
            self.fixed_point_conf = PrecisionConf(32)
        else:
            self.fixed_point_conf = fixed_point_conf

        self.public_key = public_key
        if(type(data) == np.ndarray or type(data) == TensorBase) and input_is_decrypted:
            if type(data) == np.ndarray:
                self.data = public_key.encrypt(data, True, precision_conf=self.fixed_point_conf)
            else:
                self.data = public_key.encrypt(data.data, True, precision_conf=self.fixed_point_conf)
        else:
            self.data = data

    def __add__(self, tensor):
        """Performs element-wise addition between two tensors"""

        if(not isinstance(tensor, TensorBase)):
            # try encrypting it
            tensor = PaillierTensor(self.public_key, np.array([tensor]).astype('float'), fixed_point_conf=self.fixed_point_conf)
            return PaillierTensor(self.public_key, self.data + tensor.data, False, fixed_point_conf=self.fixed_point_conf)

        if(type(tensor) == TensorBase):
            tensor = PaillierTensor(self.public_key, tensor.data, fixed_point_conf=self.fixed_point_conf)

        result_data = self.data + tensor.data
        ptensor = PaillierTensor(self.public_key, result_data, False, fixed_point_conf=self.fixed_point_conf)
        ptensor._calc_add_depth(self, tensor)
        return ptensor

    def __getitem__(self, i):
        return PaillierTensor(self.public_key, self.data[i], False)

    def __isub__(self, tensor):
        """Performs inline, element-wise subtraction between two tensors"""
        self.data -= tensor.data
        return self

    def __mul__(self, tensor):
        """Performs element-wise multiplication between two tensors"""

        if(isinstance(tensor, TensorBase)):
            if(not tensor.encrypted):
                result = self.data * tensor.data
                o = PaillierTensor(self.public_key, result, False, fixed_point_conf=self.fixed_point_conf)
                o._calc_mul_depth(self, tensor)
                return o
            else:
                return NotImplemented
        elif np.isscalar(tensor):
            # scalar is encode to match the precision of self before multiplication.
            op = self.data * tensor
            ptensor = PaillierTensor(self.public_key, op, False, fixed_point_conf=self.fixed_point_conf)
            ptensor._calc_mul_depth(self, tensor)
            return ptensor
        else:
            return NotImplemented

    def __repr__(self):
        return "PaillierTensor: " + repr(self.data)

    def __setitem__(self, key, value):
        self.data[key] = value.data
        return self

    def __str__(self):
        return "PaillierTensor: " + str(self.data)

    def __sub__(self, tensor):
        """Performs element-wise subtraction between two tensors"""
        if(not isinstance(tensor, TensorBase)):
            # try encrypting it
            tensor = PaillierTensor(self.public_key, np.array([tensor]).astype('float'))
            return PaillierTensor(self.public_key, self.data - tensor.data, False)

        if(type(tensor) == TensorBase):
            tensor = PaillierTensor(self.public_key, tensor.data)

        return PaillierTensor(self.public_key, self.data - tensor.data, False)

    def __truediv__(self, tensor):
        """Performs element-wise division between two tensors"""

        if(isinstance(tensor, TensorBase)):
            if(not tensor.encrypted):
                result = self.data * (1 / tensor.data)
                o = PaillierTensor(self.public_key, result, False, fixed_point_conf=self.fixed_point_conf)
                return o
            else:
                return NotImplemented
        elif np.isscalar(tensor):
            op = self.data * (1 / tensor)
            return PaillierTensor(self.public_key, op, False, fixed_point_conf=self.fixed_point_conf)
        else:
            return NotImplemented

    def dot(self, plaintext_x):
        if(not plaintext_x.encrypted):
            return (self * plaintext_x).sum(plaintext_x.dim() - 1)
        else:
            return NotImplemented

    def sum(self, dim=None):
        """Returns the sum of all elements in the input array."""
        if not self.encrypted:
            return NotImplemented

        if dim is None:
            return PaillierTensor(self.public_key, self.data.sum(), False, fixed_point_conf=self.fixed_point_conf)
        else:
            op = self.data.sum(axis=dim)
            return PaillierTensor(self.public_key, op, False, fixed_point_conf=self.fixed_point_conf)


class PrecisionConf:
    def __init__(self, fraction_bits=None):
        self.fraction_bits = fraction_bits

    def __eq__(self, other):
        return self.franction_bits == other.franction_bits
