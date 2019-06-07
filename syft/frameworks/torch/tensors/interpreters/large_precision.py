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
    These smaller values are of type `internal_type`. The large value is defined by `precision_fractional`

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

    def _create_internal_tensor(self):
        """Decompose a tensor into an array of numbers that represent such tensor with the required precision"""
        # TODO refs_ok might not be needed as we are passing now torch.tensors here
        # refs_ok is necessary to enable iterations of reference types.
        # These big numbers are stored as objects in np
        result = []
        for x in np.nditer(self.child, flags=["refs_ok"]):
            n = int(x.item() * self.base ** self.precision_fractional)
            if self.verbose:
                print(f"\nAdding number {n} for item {x.item()}\n")
            result.append(self._split_number(n, internal_precision[self.internal_type]))
        new_shape = self.child.shape + (len(max(result, key=len)),)
        result = np.array(result).reshape(new_shape)
        return torch.tensor(result, dtype=self.internal_type)

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response.
        """
        return {
            "internal_type": self.internal_type,
            "precision_fractional": self.precision_fractional,
        }

    @overloaded.method
    def add(self, self_, other):
        if self_.shape[-1] != other.shape[-1]:
            return self._add_different_dims(other, self_)
        else:
            return self_ + other

    @staticmethod
    def _add_different_dims(other, self_):
        """Perform addition of tensors with different shape"""
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
        self.child = self._create_internal_tensor()
        return self

    def float_precision(self):
        """
        Restore the tensor expressed now as a matrix for each original item.

        Returns:
            tensor: the original tensor.
        """
        # We need to pass the PyTorch tensor to Numpy to allow the intermediate large number.
        # An alternative would be to iterate through the PyTorch tensor and apply the restore function.
        # This however wouldn't save us from creating a new tensor
        ndarray = self.child.numpy()
        result = self._restore_tensor_into_numbers(ndarray, internal_precision[self.internal_type])
        return torch.from_numpy(result.reshape(ndarray.shape[:-1]).astype(np.float32))
        # At this point the value is an object type. Force cast to float before creating torch.tensor

    def _restore_tensor_into_numbers(self, number_array, precision):
        """Creates an numpy array containing the original numbers"""
        number_array = number_array.reshape(-1, number_array.shape[-1])
        result = []
        for elem in number_array:
            result.append(self._restore_number(elem, precision))
        return np.array(result)

    @staticmethod
    def _adjust_to_shape(to_adjust, shape, fill_value=0) -> torch.Tensor:
        """Numbers with the same precision can be represented with different matrices.
        This function adjusts the shapes of two tensors to be the same and fill the empty cells with the
        provided `fill_value`.
        We assume only the last dimension needs to be adjusted.
        Original input tensors should have the same dimensions.
        """
        diff_dim = shape[-1] - to_adjust.shape[-1]
        new_shape = to_adjust.shape[:-1] + (diff_dim,)

        if not fill_value:
            filler = torch.zeros(size=new_shape, dtype=to_adjust.dtype)
        else:
            filler = torch.ones(size=new_shape, dtype=to_adjust.dtype)
        return torch.cat([filler, to_adjust], len(shape) - 1)

    def _split_number(self, number, bits):
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
            sign = -1
            number = number * (-1)

        base = 2 ** bits
        number_parts = []
        while number:
            number, part = divmod(number, base)
            number_parts.append(part * sign)
        return number_parts[::-1]

    def _restore_number(self, number_parts, bits):
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
        return n / (self.base ** self.precision_fractional)


# The internal precision used to decompose the large numbers is half of the size of the type.
# This is because we are not considering negative numbers when decomposing.
internal_precision = {
    torch.uint8: 4,
    torch.int8: 4,
    torch.int16: 8,
    torch.short: 8,
    torch.int32: 16,
    torch.int: 16,
    torch.int64: 32,
    torch.long: 32,
}
