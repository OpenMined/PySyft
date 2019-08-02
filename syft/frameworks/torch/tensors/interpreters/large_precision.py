import numpy as np
import math
import torch

from syft.frameworks.torch.overload_torch import overloaded
from syft.frameworks.torch.tensors.interpreters import AbstractTensor


class LargePrecisionTensor(AbstractTensor):
    """LargePrecisionTensor allows handling of numbers bigger than LongTensor

    Some systems using Syft require larger types than those supported natively. This tensor type supports arbitrarily
    large values by packing them in smaller ones.
    Typically a user will require to enlarge a float number by fixing its precision
        tensor.fix_prec()

    The large value is defined by `precision_fractional`.

    The smaller values are of type `internal_type`. The split of the large number into the smaller values
    is in the range ±2**(size - 1).

    By default operations are done with NumPy. This implies unpacking the representation and packing it again.

    Check the tests to see how to play with the different parameters.
    """

    def __init__(
        self,
        owner=None,
        id=None,
        tags=None,
        description=None,
        field: int = 2 ** 512,
        base: int = 10,
        precision_fractional=0,
        internal_type=torch.int32,
        verbose=False,
    ):
        """Initializes a LargePrecisionTensor.

        Args:
            owner (BaseWorker): An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id (str or int): An optional string or integer id of the LargePrecisionTensor.
            tags (list): list of tags for searching.
            description (str): a description of this tensor.
            base (int): The base that will be used to to calculate the precision.
            precision_fractional (int): The precision required by the caller.
            internal_type (dtype): The large tensor will be stored using tensor of this type.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.field = field
        self.base = base
        self.internal_type = internal_type
        self.precision_fractional = precision_fractional
        self.verbose = verbose

    def _create_internal_representation(self):
        """Decompose a tensor into an array of numbers that represent such tensor with the required precision"""
        self_scaled = self.child.numpy() * self.base ** self.precision_fractional

        assert np.all(
            np.abs(self_scaled) < (self.field / 2)
        ), f"{self} cannot be correctly embedded: choose bigger field or a lower precision"

        # floor is applied otherwise, long float is not accurate
        self_scaled = np.vectorize(math.floor)(self_scaled)
        self_scaled %= self.field

        # self_scaled can be an array of floats. As multiplying an array of int with an int
        # still gives an array of int, I think it should be because self.child is a float tensor at this point.
        # Right now, it does not cause any problem, LargePrecisionTensor._split_numbers() returns an array of int.
        result = LargePrecisionTensor._split_numbers(
            self_scaled, self.internal_precision, self.internal_type
        )
        return torch.tensor(result, dtype=self.internal_type)

    @staticmethod
    def _expand_item(a_number, max_length):
        return [0] * (max_length - len(a_number)) + a_number

    @property
    def internal_precision(self):
        """"The internal precision used to decompose the large numbers is the size of the type - 1.
        The large number is decomposed into positive smaller numbers. This could provoke overflow if any of these
        smaller parts are bigger than type_precision/2.
        """
        return type_precision[self.internal_type] - 1

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response.
        """
        return {
            "field": self.field,
            "base": self.base,
            "internal_type": self.internal_type,
            "precision_fractional": self.precision_fractional,
        }

    @overloaded.method
    def add(self, self_, other):
        return self_ + other

    __add__ = add

    def __iadd__(self, other):
        """Add two fixed precision tensors together.
        """
        self.child = self.add(other).child

        return self

    add_ = __iadd__

    @overloaded.method
    def sub(self, self_, other):
        return self_ - other

    __sub__ = sub

    def __isub__(self, other):
        """Add two fixed precision tensors together.
        """
        self.child = self.sub(other).child

        return self

    sub_ = __isub__

    @overloaded.method
    def mul(self, self_, other):
        if isinstance(other, int):
            return self_ * other
        else:
            res = (self_ * other) % self.field

            # We need to truncate the result
            truncation = self.base ** self.precision_fractional
            gate = 1 * (res > self.field / 2)
            neg_nums = (res - self.field) // truncation + self.field
            pos_nums = res // truncation
            trunc_res = np.where(gate, neg_nums, pos_nums)

            return trunc_res

    __mul__ = mul

    def __imul__(self, other):
        self.child = self.mul(other).child
        return self

    mul_ = __imul__

    @overloaded.method
    def mod(self, self_, other):
        return self_ % other

    __mod__ = mod

    @overloaded.method
    def gt(self, self_, other):
        return 1 * (self_ > other)

    __gt__ = gt

    @overloaded.method
    def lt(self, self_, other):
        return 1 * (self_ < other)

    __lt__ = lt

    def fix_large_precision(self):
        self.child = self._create_internal_representation()
        return self

    def float_precision(self):
        """
        Restore the tensor from the internal representation.

        Returns:
            tensor: the original tensor.
        """
        result = self._internal_representation_to_large_ints()

        gate = 1 * (result > self.field / 2)
        neg_nums = (result - self.field) * gate
        pos_nums = result * (1 - gate)
        result = (neg_nums + pos_nums) / self.base ** self.precision_fractional

        # At this point the value is an object type. Force cast to float before creating torch.tensor
        return torch.from_numpy(result.reshape(self.child.shape[:-1]).astype(np.float32))

    @staticmethod
    def create_tensor_from_numpy(ndarray, **kwargs):
        """Decompose a NumPy array into an array of numbers that represent such tensor with the required precision.

        Typically this private method is called on the result of an operation.
        """
        # This method is called to rebuild an LTP after operations.
        # The wrapping is done here and not in each operation.
        ndarray %= kwargs.get("field", 2 ** 512)

        internal_type = kwargs["internal_type"]
        internal_precision = type_precision[internal_type] - 1

        result = LargePrecisionTensor._split_numbers(ndarray, internal_precision, internal_type)
        return torch.tensor(result, dtype=internal_type)

    @staticmethod
    def _split_numbers(numbers, bits, internal_type) -> np.array:
        """Splits a tensor of numbers in numbers of a smaller power.

        Args:
            numbers (array): the tensor to split.
            bits (int): the bits to use in the split.

        Returns:
            array: a tensor with one more dimension representing the original one.

        """
        if np.all(numbers == 0):
            # numbers is an array of objects if the values are too large
            # we need to cast it back to an array of integers
            numbers = numbers.astype(np.int)
            return np.expand_dims(numbers, -1)

        sign_mask = np.where(numbers > 0, 1, -1)
        if internal_type == torch.uint8:
            assert np.all(
                sign_mask == 1
            ), "LargePrecisionTensors with negative values cannot be represented with uint8"
        numbers = np.where(numbers > 0, numbers, -numbers)

        base = 2 ** bits
        number_parts = []
        while np.any(numbers):
            # number, part = np.divmod(number, base)  # Not sure why this doesn't work
            part = numbers % base
            numbers = numbers // base
            number_parts.append(part * sign_mask)

        res = np.array(number_parts[::-1], dtype=np.int)
        return res.transpose(*range(1, res.ndim), 0)

    def _internal_representation_to_large_ints(self) -> np.array:
        """Creates an numpy array containing the objective large numbers."""
        ndarray = self.child.numpy()
        ndarray = ndarray.reshape(-1, ndarray.shape[-1])
        result = []
        for elem in ndarray:
            result.append(LargePrecisionTensor._restore_large_number(elem, self.internal_precision))
        return np.array(result).reshape(self.child.shape[:-1])

    @staticmethod
    def _restore_large_number(number_parts, bits) -> int:
        """Rebuilds a number from a numpy array.

        Args:
            number_parts (ndarray): the numpy array of numbers representing the original one.
            bits (int): the bits used in the split.

        Returns:
            Number: the large number represented by this tensor
        """

        def _restore_recursive(parts, acc, base):
            if len(parts) == 0:
                return acc
            return _restore_recursive(parts[1:], acc * base + parts[0].item(), base)

        return _restore_recursive(number_parts, 0, 2 ** bits)


# The size of each type
type_precision = {
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
}
