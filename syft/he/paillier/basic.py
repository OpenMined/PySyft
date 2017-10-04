import numpy as np
import pickle
from ...tensor import TensorBase
import copy


class PaillierTensor(TensorBase):

    def __init__(self, public_key, data=None, input_is_decrypted=True, fixed_point_conf=None):
        self.encrypted = True
        if fixed_point_conf is None:
            self.fixed_point_conf = FixedPointConfig()
        else:
            self.fixed_point_conf = fixed_point_conf

        self.public_key = public_key
        if(type(data) == np.ndarray or type(data) == TensorBase) and input_is_decrypted:
            if type(data) == np.ndarray:
                self.data = public_key.encrypt(data, True, self.fixed_point_conf)
            else:
                self.data = public_key.encrypt(data.data, True, self.fixed_point_conf)
        else:
            self.data = data

    def __setitem__(self, key, value):
        self.data[key] = value.data
        return self

    def __getitem__(self, i):
        return PaillierTensor(self.public_key, self.data[i], False, self.fixed_point_conf)

    def __add__(self, tensor):
        """Performs element-wise addition between two tensors"""

        if(not isinstance(tensor, TensorBase)):
            # try encrypting it
            tensor = PaillierTensor(self.public_key, np.array([tensor]).astype('float'), fixed_point_conf=self.fixed_point_conf)
            return PaillierTensor(self.public_key, self.data + tensor.data, False, self.fixed_point_conf)

        if(type(tensor) == TensorBase):
            tensor = PaillierTensor(self.public_key, tensor.data, fixed_point_conf=self.fixed_point_conf)

        result_data = self.data + tensor.data
        ptensor = PaillierTensor(self.public_key, result_data, False, result_data.flat[0].config)
        ptensor._calc_add_depth(self, tensor)
        return ptensor

    def __sub__(self, tensor):
        """Performs element-wise subtraction between two tensors"""

        if(not isinstance(tensor, TensorBase)):
            # try encrypting it
            tensor = PaillierTensor(self.public_key, np.array([tensor]).astype('float'), fixed_point_conf=self.fixed_point_conf)
            return PaillierTensor(self.public_key, self.data - tensor.data, False, self.fixed_point_conf)

        if(type(tensor) == TensorBase):
            tensor = PaillierTensor(self.public_key, tensor.data, fixed_point_conf=self.fixed_point_conf)

        result_data = self.data - tensor.data
        ptensor = PaillierTensor(self.public_key, result_data, False, result_data.flat[0].config)
        return ptensor

    def __isub__(self, tensor):
        """Performs inline, element-wise subtraction between two tensors"""
        self.data -= tensor.data
        return self

    def __mul__(self, tensor):
        """Performs element-wise multiplication between two tensors"""

        if(isinstance(tensor, TensorBase)):
            if(not tensor.encrypted):
                result = self.data * tensor.data
                o = PaillierTensor(self.public_key, result, False, self.fixed_point_conf)
                o._calc_mul_depth(self, tensor)
                return o
            else:
                return NotImplemented
        else:
            op = self.data * float(tensor)
            ptensor = PaillierTensor(self.public_key, op, False, self.fixed_point_conf)
            ptensor._calc_mul_depth(self, tensor)
            return ptensor

    def __truediv__(self, tensor):
        """Performs element-wise division between two tensors"""

        if(isinstance(tensor, TensorBase)):
            if(not tensor.encrypted):
                result = self.data * (1 / tensor.data)
                o = PaillierTensor(self.public_key, result, False, self.fixed_point_conf)
                return o
            else:
                return NotImplemented
        else:
            op = self.data * (1 / float(tensor))
            return PaillierTensor(self.public_key, op, False, self.fixed_point_conf)

    def sum(self, dim=None):
        """Returns the sum of all elements in the input array."""
        if not self.encrypted:
            return NotImplemented

        if dim is None:
            return PaillierTensor(self.public_key, self.data.sum(), False, self.fixed_point_conf)
        else:
            op = self.data.sum(axis=dim)
            return PaillierTensor(self.public_key, op, False, self.fixed_point_conf)

    def dot(self, plaintext_x):
        if(not plaintext_x.encrypted):
            return (self * plaintext_x).sum(plaintext_x.dim() - 1)
        else:
            return NotImplemented

    def __str__(self):
        return "PaillierTensor: " + str(self.data)

    def __repr__(self):
        return "PaillierTensor: " + repr(self.data)


class FixedPointConfig(object):
    def __init__(self, base=10, precision_integral=8, precision_fractional=8,
                 big_prime=293973345475167247070445277780365744413):
        self.BASE = base
        self.PRECISION_INTEGRAL = precision_integral
        self.PRECISION_FRACTIONAL = precision_fractional
        self.Q = big_prime
        assert(self.Q > self.BASE**(self.PRECISION_INTEGRAL + self.PRECISION_FRACTIONAL))


class FixedPoint:

    def __init__(self, public_key, data=None, config=None):
        """Wraps pointer to encrypted FixedPoint with an interface that numpy
        can use."""
        if config is None:
            self.config = FixedPointConfig()
        else:
            self.config = config

        self.public_key = public_key
        if data is not None:
            data_fixed_point = self.encode(data)
            self.data = self.public_key.pk.encrypt(data_fixed_point)
        else:
            self.data = None

    def decrypt(self, secret_key):
        return self.decode(secret_key.decrypt(self.data))

    def __add__(self, y):
        """Adds two encrypted FixedPoints together."""
        shift = abs(y.config.PRECISION_FRACTIONAL - self.config.PRECISION_FRACTIONAL)
        if self.config.PRECISION_FRACTIONAL < y.config.PRECISION_FRACTIONAL:
            data_decimal_aligned = self.decrease_decimal_places(shift_by=shift)
            out = FixedPoint(self.public_key, None, config=data_decimal_aligned.config)
            out.data = data_decimal_aligned.data + y.data
            return out
        elif self.config.PRECISION_FRACTIONAL > y.config.PRECISION_FRACTIONAL:
            data_decimal_aligned = y.decrease_decimal_places(shift_by=shift)
            out = FixedPoint(self.public_key, None, config=data_decimal_aligned.config)
            out.data = data_decimal_aligned.data + self.data
            return out
        else:
            out = FixedPoint(self.public_key, None, config=self.config)
            out.data = self.data + y.data
            return out

    def __sub__(self, y):
        """Subtracts two encrypted FixedPoints."""
        shift = abs(y.config.PRECISION_FRACTIONAL - self.config.PRECISION_FRACTIONAL)
        if self.config.PRECISION_FRACTIONAL < y.config.PRECISION_FRACTIONAL:
            data_decimal_aligned = self.decrease_decimal_places(shift_by=shift)
            out = FixedPoint(self.public_key, None, config=data_decimal_aligned.config)
            out.data = data_decimal_aligned.data - y.data
            return out
        elif self.config.PRECISION_FRACTIONAL > y.config.PRECISION_FRACTIONAL:
            data_decimal_aligned = y.decrease_decimal_places(shift_by=shift)
            out = FixedPoint(self.public_key, None, config=data_decimal_aligned.config)
            out.data = self.data - data_decimal_aligned.data
            return out
        else:
            out = FixedPoint(self.public_key, None, config=self.config)
            out.data = self.data - y.data
        return out

    def __mul__(self, y):
        """Multiplies two FixedPoints. y may be encrypted or a simple FixedPoint."""

        if(type(y) == type(self)):
            out = FixedPoint(self.public_key, None, config=self.config)
            out.data = self.data * y.data
            return out
        elif(type(y) == int or type(y) == float):
            out = FixedPoint(self.public_key, None, config=self.config)
            out.data = self.data * y
            return out
        else:
            return None

    def __truediv__(self, y):
        """Divides two FixedPoints. y may be encrypted or a simple FixedPoint."""

        if(type(y) == type(self)):
            out = FixedPoint(self.public_key, None, config=self.config)
            out.data = self.data / y.data
            return out
        elif(type(y) == int):
            out = FixedPoint(self.public_key, None, config=self.config)
            out.data = self.data / y
            return out
        else:
            return None

    def __repr__(self):
        """This is kindof a boring/uninformative __repr__"""

        return 'e'

    def __str__(self):
        """This is kindof a boring/uninformative __str__"""

        return 'e'

    def encode(self, rational):
        """encodes the rational input to a natural number, so it is compatible Paillier encryption"""
        upscaled = int(rational * self.config.BASE ** self.config.PRECISION_FRACTIONAL)
        field_element = upscaled % self.config.Q
        return field_element

    def decode(self, field_element):
        """decodes the `field_element` (which is natural number) to a rational number"""
        upscaled = field_element if field_element <= self.config.Q / 2 else field_element - self.config.Q
        rational = upscaled / self.config.BASE ** self.config.PRECISION_FRACTIONAL
        return rational

    def decrease_decimal_places(self, shift_by):
        out = FixedPoint(self.public_key, None, config=copy.copy(self.config))
        out.data = self.data * (self.config.BASE ** shift_by)
        out.config.PRECISION_FRACTIONAL += shift_by
        out.config.PRECISION_INTEGRAL -= shift_by
        return out

    def serialize(self):
        return pickle.dumps(self)

    def deserialize(b):
        return pickle.loads(b)
