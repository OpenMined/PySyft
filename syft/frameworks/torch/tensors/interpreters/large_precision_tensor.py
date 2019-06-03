import numpy as np
import torch

from syft.frameworks.torch.tensors.interpreters import AbstractTensor


class LargePrecisionTensor(AbstractTensor):
    """LargePrecisionTensor allows handling of numbers bigger than LongTensor

    Some systems using Syft require larger types than those supported natively. This tensor type supports arbitrarily
    large values by packing them in smaller ones.
    If the original tensor is a PyTorch tensor it is not processed
    """

    def __init__(self, owner=None, id=None, tags=None, description=None, precision=16, virtual_prec=0):
        """Initializes a LargePrecisionTensor.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LargePrecisionTensor.
            precision: The precision this tensor will be transformed to internally. In bits.
            virtual_prec: The virtual precision required by the caller. In bits.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        self.precision = precision
        self.virtual_prec = virtual_prec

    @staticmethod
    def _create_internal_tensor(tensor, precision, virtual_prec):
        # TODO refs_ok might not be needed as we are passing now torch.tensors here
        # refs_ok is necessary to enable iterations of reference types.
        # These big numbers are stored as objects in np
        result = []
        for x in np.nditer(tensor, flags=["refs_ok"]):
            n = int(x.item() * 2 ** virtual_prec)
            print(f"\nAdding number {n} for item {x.item()}\n")
            result.append(LargePrecisionTensor._split_number(n, precision))
        new_shape = tensor.shape + (len(max(result, key=len)),)
        result = np.array(result).reshape(new_shape)
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
        if self.shape[-1] != other.shape[-1]:
            if self.shape[-1] < other.shape[-1]:
                result = self._adjust_to_shape(other.shape) + other.child
            else:
                result = other._adjust_to_shape(self.shape) + self.child
        else:
            result = self.child + other.child
        return LargePrecisionTensor(tensor=result, precision=self.precision)

    def _adjust_to_shape(self, shape, fill_value=0) -> torch.Tensor:
        # We assume only the last dimension needs to be adjusted
        # Original input tensors should have the same dimensions
        diff_dim = shape[-1] - self.shape[-1]
        new_shape = self.shape[:-1] + (diff_dim,)
        if not fill_value:
            filler = torch.zeros(size=new_shape, dtype=self.child.dtype)
        else:
            filler = torch.ones(size=new_shape, dtype=self.child.dtype)
        result = torch.cat([filler, self.child], len(shape) - 1)
        return result

    @staticmethod
    def _split_number(number, bits):
        """Splits a number in numbers of a smaller power

        :param number: the number to split
        :param bits: the bits to use in the split
        :return: a list of numbers representing the original one
        """
        if not number:
            return [number]
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

    @staticmethod
    def _restore_number_np(number_parts, bits):
        """Rebuild a number from a numpy array

        :param number_parts: the numpy array of numbers representing the original one
        :param bits: the bits used in the split
        :return: the original number
        """
        base = 2 ** bits
        n = 0
        # Using tolist() to allow int type. Terrible :(
        for x in number_parts.tolist():
            n = n * base + x
        return n

    def fix_large_precision(self):
        self.child = self._create_internal_tensor(self.child, self.precision, self.virtual_prec)
        return self

    def restore_precision(self):
        """
        Restore the tensor expressed now as a matrix for each original item
        :return: the original tensor
        """
        # TODO Is Vectorization possible here?
        return torch.from_numpy((np.apply_along_axis(lambda x: self._restore_number_np(x, self.precision),
                                1,
                                self.child.numpy()) / (2 ** self.virtual_prec))
                                .astype(np.float)
                                )
