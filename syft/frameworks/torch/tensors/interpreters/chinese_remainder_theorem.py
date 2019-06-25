import torch
import math
import itertools

from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.overload_torch import overloaded


class CRTTensor(AbstractTensor):
    """ A CRT tensor is a tensor whose values are represented as their remainders modulo several pairwise coprime numbers.
    The true tensor values lie in the Zq field where q is the product of all the modulos.
    A CRT tensor then represent a modular system of equations:
        x = a_0 mod f_0
        ...
        x = a_n mod f_n
    The Chineses Remainder Theorem (that also gives his name to this tensor) assert that exactly one x in Zq with q = f_0 * ... * f_n
    satisfies this system. This x is the real value represented by the tensor.
    This tensor makes it possible to represent big numbers and to avoid overflows while manipulating them.
    It also makes additions, subtractions, and multiplications of huge numbers quite efficient.
    """

    def __init__(
        self, residues: dict = None, field=None, owner=None, id=None, tags=None, description=None
    ):
        super().__init__(owner=owner, id=id, tags=tags, description=description)

        # check that all the modulos are pairwise coprime
        for pair in itertools.combinations(residues.keys(), r=2):
            assert (
                math.gcd(pair[0], pair[1]) == 1
            ), f"{pair[0]} and {pair[1]} are not coprime, you cannot build a CRTTensor with these as modulos"

        prod_modulo = 1
        res_shape = next(iter(residues.values())).shape
        self.child = {}
        for f, r in residues.items():
            assert isinstance(r.child, FixedPrecisionTensor)
            assert f == r.child.field
            self.child[f] = r
            prod_modulo *= f
            assert r.shape == res_shape, "All residue tensors of CRTTensor must have the same shape"

        if field is not None:
            assert prod_modulo == field
        else:
            field = prod_modulo

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        for mod, res in self.child.items():
            out += f"\n\t {mod} -> " + str(res)
        return out

    def __repr__(self):
        return self.__str__()

    @overloaded.method
    def add(self, self_, other):
        assert self_.keys() == other.keys(), "Cannot add 2 CRT that don't have the same modulos"

        result = {}

        for mod in self_.keys():
            result[mod] = self_[mod] + other[mod]

        return result

    __add__ = add

    @overloaded.method
    def sub(self, self_, other):
        assert (
            self_.keys() == other.keys()
        ), "Cannot subtract 2 CRT that don't have the same modulos"

        result = {}

        for mod in self_.keys():
            result[mod] = self_[mod] - other[mod]

        return result

    __sub__ = sub

    @overloaded.method
    def mul(self, self_, other):
        assert (
            self_.keys() == other.keys()
        ), "Cannot multiply 2 CRT that don't have the same modulos"

        result = {}

        for mod in self_.keys():
            result[mod] = self_[mod] * other[mod]

        return result

    __mul__ = mul

    def reconctruct(self):
        """ Build the tensor in self.field satisfying the modular system represented by self
        """
        pass
