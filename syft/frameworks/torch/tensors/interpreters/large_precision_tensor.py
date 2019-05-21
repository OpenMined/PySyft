import torch

from syft.frameworks.torch.tensors.interpreters import AbstractTensor


class LargePrecisionTensor(AbstractTensor):
    """LargePrecisionTensor allows handling of numbers bigger than LongTensor

    Some systems using Syft require larger types than those supported natively. This tensor type supports arbitrarily
    large values by packing them in smaller ones.

    """

    def __init__(self, tensor, owner=None, id=None, tags=None, description=None, to_bits=32):
        """Initializes a LargePrecisionTensor.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LargePrecisionTensor.
            to_bits: The objective precision for this tensor
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)
        assert to_bits % 2 == 0, "%r is not power of two" % to_bits
        self.precision = to_bits
        # TODO Start with a single dim
        self.child = torch.IntTensor(self._split_number(tensor[0], to_bits))

    def __eq__(self, other):
        return self.child == other.child

    # TODO Having issues with the hook
    # @overloaded.method
    def add(self, other, **kwargs):
        # TODO Check types, precision
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
