import torch
import numpy as np
from syft.frameworks.torch.overload_torch import overloaded
from syft.frameworks.torch.tensors.interpreters import AbstractTensor


class LargePrecisionTensor(AbstractTensor):
    """LargePrecisionTensor allows handling of numbers bigger than LongTensor

    Some systems using Syft require larger types than those supported natively. This tensor type supports arbitrarily
    large values by packing them in smaller ones.
    Note that the original tensor cannot be a PyTorch tensor as it would cause a RuntimeError if the number is too large
    """

    def __init__(self, tensor, owner=None, id=None, tags=None, description=None, precision=32):
        """Initializes a LargePrecisionTensor.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LargePrecisionTensor.
            precision: The objective precision for this tensor
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        assert precision % 2 == 0, "%r is not power of two" % precision
        self.precision = precision
        # torch.IntTensor(self._split_number(tensor[0], precision))
        self.child = self._create_internal_tensor(tensor, precision)

    @staticmethod
    def _create_internal_tensor(tensor, precision):
        # refs_ok is necessary to enable iterations of reference types.
        # These big numbers are stored as objects in np
        result = []
        for x in np.nditer(tensor, flags=["refs_ok"]):
            result.append(LargePrecisionTensor._split_number(x.item(), precision))
        new_shape = tensor.shape + (len(max(result, key=len)),)
        result = np.array(result).reshape(new_shape)
        # TODO Note Assuming all results are of the same shape. This would be hardly the case
        return torch.IntTensor(result)

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response.
        """
        return {
            "field": self.precision,
        }

    def __eq__(self, other):
        return self.child == other.child

    # TODO Having issues with the hook
    # @overloaded.method
    def add(self, other):
        assert isinstance(other, LargePrecisionTensor), "LargePrecisionTensor cannot be added to %r" % type(other)
        assert self.child.shape == other.child.shape,\
            "Original numbers generated tensors of different shapes %r %r" % self.child.shape % other.child.shape
        return self.child + other.child

    @staticmethod
    def _split_number(number, bits):
        """Splits a number in numbers of a smaller power

        :param number: the number to split
        :param bits: the bits to use in the split
        :return: a list of numbers representing the original one
        """
        base = 2 ** bits
        number_parts = []
        while number:
            number, part = divmod(number, base)
            number_parts.append(part)
        return number_parts[::-1]

    @staticmethod
    def _restore_number(number_parts, bits):
        """Rebuild a number from its parts

        :param number_parts: the list of numbers representing the original one
        :param bits: the bits used in the split
        :return: the original number
        """
        base = 2 ** bits
        n = 0
        number_parts = number_parts[::-1]
        while number_parts:
            n = n * base + number_parts.pop()
        return n
