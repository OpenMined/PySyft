import numpy as np
import torch

from syft.frameworks.torch.overload_torch import overloaded
from syft.frameworks.torch.tensors.interpreters import AbstractTensor


class LargePrecisionTensor(AbstractTensor):
    """LargePrecisionTensor allows handling of numbers bigger than LongTensor

    Some systems using Syft require larger types than those supported natively. This tensor type supports arbitrarily
    large values by packing them in smaller ones.
    Typically a user will require to enlarge a float number by fixing its precision
        tensor.fix_prec()
    These smaller values are of type `internal_type`. The large value is defined by `precision_fractional`.

    By default operations are done with NumPy. This implies unpacking the representation and packing it again.

    Check the tests to see how to play with the different parameters.
    """

    def __init__(
        self,
        owner=None,
        id=None,
        tags=None,
        description=None,
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
        self.base = base
        self.internal_type = internal_type
        self.precision_fractional = precision_fractional
        self.verbose = verbose

    def _create_internal_representation(self):
        """Decompose a tensor into an array of numbers that represent such tensor with the required precision"""
        result = []
        for x in np.nditer(self.child):
            n = int(x.item() * self.base ** self.precision_fractional)
            if self.verbose:
                print(f"\nAdding number {n} for item {x.item()}\n")
            result.append(self._split_number(n, internal_precision[self.internal_type]))
        new_shape = self.child.shape + (len(max(result, key=len)),)
        result = np.array(result).reshape(new_shape)
        return torch.tensor(result, dtype=self.internal_type)

    def _create_tensor_from_numpy(self, ndarray):
        """Decompose a NumPy array into an array of numbers that represent such tensor with the required precision.

        Typically this private method is called on the result of an operation.
        """
        result = []
        # refs_ok is necessary to enable iterations of reference types.
        for x in np.nditer(ndarray, flags=["refs_ok"]):
            result.append(self._split_number(x.item(), internal_precision[self.internal_type]))
        new_shape = self.child.shape[:-1] + (len(max(result, key=len)),)
        result = np.array(result).reshape(new_shape)
        return torch.tensor(result, dtype=self.internal_type)

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response.
        """
        return {
            "base": self.base,
            "internal_type": self.internal_type,
            "precision_fractional": self.precision_fractional,
        }

    @overloaded.method
    def add(self, self_, other):
        a = LargePrecisionTensor._internal_representation_to_large_ints(
            self_, internal_precision[self.internal_type]
        )
        b = LargePrecisionTensor._internal_representation_to_large_ints(
            other, internal_precision[self.internal_type]
        )
        return self._create_tensor_from_numpy(a + b)

    __add__ = add

    def __iadd__(self, other):
        """Add two fixed precision tensors together.
        """
        self.child = self.add(other).child

        return self

    @overloaded.method
    def sub(self, self_, other):
        a = LargePrecisionTensor._internal_representation_to_large_ints(
            self_, internal_precision[self.internal_type]
        )
        b = LargePrecisionTensor._internal_representation_to_large_ints(
            other, internal_precision[self.internal_type]
        )
        return self._create_tensor_from_numpy(a - b)

    __sub__ = sub

    def __isub__(self, other):
        """Add two fixed precision tensors together.
        """
        self.child = self.sub(other).child

        return self

    @overloaded.method
    def mul(self, self_, other):
        a = LargePrecisionTensor._internal_representation_to_large_ints(
            self_, internal_precision[self.internal_type]
        )
        b = LargePrecisionTensor._internal_representation_to_large_ints(
            other, internal_precision[self.internal_type]
        )
        # We need to divide the result of the multiplication by the precision
        return self._create_tensor_from_numpy((a * b) / (self.base ** self.precision_fractional))

    __mul__ = mul

    def fix_large_precision(self):
        self.child = self._create_internal_representation()
        return self

    def float_precision(self):
        """
        Restore the tensor from the internal representation.

        Returns:
            tensor: the original tensor.
        """
        result = self._internal_representation_to_large_ints(
            self.child, internal_precision[self.internal_type]
        )
        result /= self.base ** self.precision_fractional
        # At this point the value is an object type. Force cast to float before creating torch.tensor
        return torch.from_numpy(result.reshape(self.child.shape[:-1]).astype(np.float32))

    def _split_number(self, number, bits) -> list:
        """Splits a number in numbers of a smaller power.

        Args:
            number (int): the number to split.
            bits (int): the bits to use in the split.

        Returns:
            list: a list of numbers representing the original one.

        """
        if not number:
            return [number]

        if number > 0:
            sign = 1
        else:
            assert (
                self.internal_type != torch.uint8
            ), "Negative LargePrecisionTensors cannot be represented with uint8"
            sign = -1
            number = number * (-1)

        base = 2 ** bits
        number_parts = []
        while number:
            number, part = divmod(number, base)
            number_parts.append(part * sign)
        return number_parts[::-1]

    @staticmethod
    def _internal_representation_to_large_ints(tensor, precision) -> np.array:
        """Creates an numpy array containing the objective large numbers."""
        ndarray = tensor.numpy()
        ndarray = ndarray.reshape(-1, ndarray.shape[-1])
        result = []
        for elem in ndarray:
            result.append(LargePrecisionTensor._restore_large_number(elem, precision))
        return np.array(result)

    @staticmethod
    def _restore_large_number(number_parts, bits) -> int:
        """Rebuilds a number from a numpy array.

        Args:
            number_parts (ndarray): the numpy array of numbers representing the original one.
            bits (int): the bits used in the split.

        Returns:
            Number: the large number represented by this tensor
        """
        base = 2 ** bits
        n = 0
        for x in number_parts.tolist():
            n = n * base + x
        return n


# The internal precision used to decompose the large numbers is the size of the type - 1.
internal_precision = {
    torch.uint8: 7,
    torch.int8: 7,
    torch.int16: 15,
    torch.short: 15,
    torch.int32: 31,
    torch.int: 31,
    torch.int64: 63,
    torch.long: 63,
}
