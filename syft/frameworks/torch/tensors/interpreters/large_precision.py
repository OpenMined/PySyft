import numpy as np
import torch

from syft.frameworks.torch.overload_torch import overloaded
from syft.frameworks.torch.tensors.interpreters import AbstractTensor


class LargePrecisionTensor(AbstractTensor):
    """LargePrecisionTensor allows handling of numbers bigger than LongTensor

    Some systems using Syft require larger types than those supported natively. This tensor type supports arbitrarily
    large values by packing them in smaller ones.
    """

    def __init__(
        self, owner=None, id=None, tags=None, description=None, precision=16, virtual_prec=0
    ):
        """Initializes a LargePrecisionTensor.

        Args:
            owner (BaseWorker): An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id (str or int): An optional string or integer id of the LargePrecisionTensor.
            tags (list): list of tags for searching.
            description (str): a description of this tensor.
            precision (int): The precision this tensor will be transformed to internally. In bits.
            virtual_prec (int): The virtual precision required by the caller. In bits.
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
        return {"precision": self.precision, "virtual_prec": self.virtual_prec}

    def __eq__(self, other):
        return self.child == other.child

    @overloaded.method
    def add(self, self_, other):
        if self_.shape[-1] != other.shape[-1]:
            return self._add_different_dims(other, self_)
        else:
            return self_ + other

    @staticmethod
    def _add_different_dims(other, self_):
        if self_.shape[-1] < other.shape[-1]:
            result = LargePrecisionTensor._adjust_to_shape(self_, other.shape) + other
        else:
            result = LargePrecisionTensor._adjust_to_shape(other, self_.shape) + self_
        return result

    __add__ = add

    def __iadd__(self, other):
        """Add two fixed precision tensors together.
        """
        self.child = self.add(other).child

        return self

    def fix_large_precision(self):
        self.child = self._create_internal_tensor(self.child, self.precision, self.virtual_prec)
        return self

    def restore_precision(self):
        """
        Restore the tensor expressed now as a matrix for each original item.

        Returns:
            tensor: the original tensor.
        """
        # We need to pass the PyTorch tensor to Numpy to allow the intermediate large number.
        # An alternative would be to iterate through the PyTorch tensor and apply the restore function.
        # This however wouldn't save us from creating a new tensor
        ndarray = self.child.numpy()
        result = LargePrecisionTensor._restore_tensor_into_numbers(ndarray, self.precision) / (
            2 ** self.virtual_prec
        )
        return torch.from_numpy(result.reshape(ndarray.shape[:-1]).astype(np.float32))
        # At this point the value is an object type. Force cast to float before creating torch.tensor

    @staticmethod
    def _restore_tensor_into_numbers(number_array, precision):
        number_array = number_array.reshape(-1, number_array.shape[-1])
        result = []
        for elem in number_array:
            result.append(LargePrecisionTensor._restore_number(elem, precision))
        return np.array(result)

    @staticmethod
    def _adjust_to_shape(to_adjust, shape, fill_value=0) -> torch.Tensor:
        # We assume only the last dimension needs to be adjusted
        # Original input tensors should have the same dimensions
        diff_dim = shape[-1] - to_adjust.shape[-1]
        new_shape = to_adjust.shape[:-1] + (diff_dim,)

        if not fill_value:
            filler = torch.zeros(size=new_shape, dtype=to_adjust.dtype)
        else:
            filler = torch.ones(size=new_shape, dtype=to_adjust.dtype)
        return torch.cat([filler, to_adjust], len(shape) - 1)

    @staticmethod
    def _split_number(number, bits):
        """Splits a number in numbers of a smaller power.

        Args:
            number (int): the number to split.
            bits (int): the bits to use in the split.

        Returns:
            list: a list of numbers representing the original one.

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
        """Rebuild a number from a numpy array,

        Args:
            number_parts (ndarray): the numpy array of numbers representing the original one.
            bits (int): the bits used in the split.

        Returns:
            Number: the original number.
        """
        base = 2 ** bits
        n = 0
        # Using tolist() to allow int type. Terrible performance :(
        for x in number_parts.tolist():
            n = n * base + x
        return n
