import torch
import math
import itertools

import syft
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.precision import FixedPrecisionTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.overload_torch import overloaded


class CRTTensor(AbstractTensor):
    """ A CRT tensor is a tensor whose values are represented as their remainders modulo several pairwise coprime numbers.
    The true tensor values lie in the Zq field where q is the product of all the moduli.
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
        self,
        field: int = 21,
        residues: dict = None,
        base=None,
        precision_fractional=None,
        reconstruction_coeffs=None,
        owner=None,
        id=None,
        tags: set = None,
        description: str = None,
    ):
        super().__init__(owner=owner, id=id, tags=tags, description=description)

        # Check that all the moduli are pairwise coprime
        if residues is not None:
            for pair in itertools.combinations(residues.keys(), r=2):
                assert (
                    math.gcd(pair[0], pair[1]) == 1
                ), f"{pair[0]} and {pair[1]} are not coprime, you cannot build a CRTTensor with these as moduli"

        self.field = field

        # Check that all the residues have the same precision and
        # check that all the shapes are the same
        # while filling tensor
        self.base = base
        self.precision_fractional = precision_fractional
        if residues is not None:
            r = next(iter(residues.values()))  # Take one arbitrary residue
            assert isinstance(
                r.child, FixedPrecisionTensor
            ), "To build a CRTTensor directly, the residue argument should be a dictionary \
                where keys are moduli and values are the residues under the form of FixedPrecisionTensor"
            b = r.child.base
            prec_frac = r.child.precision_fractional
            res_shape = r.shape
            self.child = {}
            for f, r in residues.items():
                assert isinstance(
                    r.child, FixedPrecisionTensor
                ), "To build a CRTTensor directly, the residue argument should be a dictionary \
                    where keys are moduli and values are the residues under the form of FixedPrecisionTensor"
                assert r.child.base == b, "All residue tensors of CRTTensor must have the same base"
                assert (
                    r.child.precision_fractional == prec_frac
                ), "All residue tensors of CRTTensor must have the same precision_fractional"
                assert (
                    f == r.child.field
                ), "All residue tensors of CRTTensor must have the same precision_fractional"
                assert (
                    r.shape == res_shape
                ), "All residue tensors of CRTTensor must have the same shape"

                self.child[f] = r

            if base is not None:
                assert b == base
            if precision_fractional is not None:
                assert prec_frac == precision_fractional
            self.base = b
            self.precision_fractional = prec_frac

        # In order to avoid computing the coefficients to solve the modular system several times,
        # we can store them once computed. See the docstring of the solve_system method for more details.
        if residues is not None and reconstruction_coeffs is None:
            reconstruction_coeffs = self.compute_reconstruction_coeffs()
        self.reconstruction_coeffs = reconstruction_coeffs

    def to_crt_representation(self):
        """This method encodes the .child object to CRT representation """

        fix_prec_tensor = self.child
        self.base = fix_prec_tensor.child.base
        self.precision_fractional = fix_prec_tensor.child.precision_fractional

        try:
            moduli = _moduli_for_fields[fix_prec_tensor.child.field]
        except KeyError as e:
            possible_moduli = list(_moduli_for_fields.keys())
            raise Exception(
                f"Only tensor in fields of size in {possible_moduli} are currently possible to represent with CRT tensors"
            ) from e

        self.child = {}
        for mod in moduli:
            self.child[mod] = fix_prec_tensor % mod

        self.reconstruction_coeffs = self.compute_reconstruction_coeffs()

        return self

    def float_precision(self):
        reconstructed = self.reconstruct()
        return reconstructed.float_precision()

    @property
    def grad(self):
        """ Gradient makes no sense for CRT Tensor, so we make it clear
        that if someone query .grad on a CRT Tensor it doesn't error
        but returns grad and can't be set
        """
        return None

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        for mod, res in self.child.items():
            out += f"\n\t -> {mod}: {res}"
        return out

    def __repr__(self):
        return self.__str__()

    @property
    def shape(self) -> torch.Size:
        return next(iter(self.child.values())).shape

    @overloaded.method
    def eq(self, self_, other_):
        assert (
            self_.keys() == other_.keys()
        ), "Cannot compare 2 CRT tensors that don't have the same moduli"

        result = torch.ones(self.shape).fix_prec(
            base=self.base, precision_fractional=self.precision_fractional
        )
        for mod in self_.keys():
            result *= (self_[mod] == other_[mod]).long()

        dict_result = {}
        for mod in self_.keys():
            # This is a bit ugly, I want the result to be stored in a CRT tensor with same moduli
            result.child.field = mod
            dict_result[mod] = result.copy()
        return dict_result

    __eq__ = eq

    def __neg__(self):
        return -1 * self

    @overloaded.method
    def add(self, self_, other):

        result = {}

        if isinstance(other, int):
            for mod in self_.keys():
                result[mod] = self_[mod] + other

        else:
            assert (
                self_.keys() == other.keys()
            ), "Cannot add 2 CRT tensors that don't have the same moduli"

            for mod in self_.keys():
                result[mod] = self_[mod] + other[mod]

        return result

    __add__ = add
    __radd__ = add

    @overloaded.method
    def sub(self, self_, other):

        result = {}

        if isinstance(other, int):
            for mod in self_.keys():
                result[mod] = self_[mod] - other

        else:
            assert (
                self_.keys() == other.keys()
            ), "Cannot subtract 2 CRT tensors that don't have the same moduli"

            for mod in self_.keys():
                result[mod] = self_[mod] - other[mod]

        return result

    __sub__ = sub

    def __rsub__(self, other):
        return (-self).sub(-other)

    @overloaded.method
    def mul(self, self_, other):

        result = {}

        if isinstance(other, int):
            for mod in self_.keys():
                result[mod] = self_[mod] * other

        else:
            assert (
                self_.keys() == other.keys()
            ), "Cannot multiply 2 CRT tensors that don't have the same moduli"

            for mod in self_.keys():
                result[mod] = self_[mod] * other[mod]

        return result

    __mul__ = mul
    __rmul__ = mul

    def div(self, other):
        raise NotImplementedError

    __truediv__ = div

    def reconstruct(self):
        """ Build the tensor in Zq with q = prod(self.child.keys())
        satisfying the modular system represented by the tensor.
        The algorithm consists in:
        1) Compute N = prod(self.child.keys())
        2) yi = N / ni where the ni's are moduli in the system
        3) zi = yi^(-1) mod ni (we know zi exists because all the original moduli are pairwise coprime)
        4) The result is x = sum(ai * yi * zi) where ai is the residue modulo ni in the system
        
        We can see that this boils down to a linear combination of ai's with coefficients yi * zi.
        These coefficients depend only on the moduli used to represent the tensor so we don't need
        to compute them several times for the same tensor. We can also reuse them for a tensor produced
        by operations on self because 1) the operations can only be done between tensors with the same moduli
        and 2) the output tensor will have the same moduli.
        This is why we store them after having computed them once.
        """
        res = 0
        N = 1
        # We need to sort the dict to be sure that the coeffs are retrieved in the same order they were stored
        for i, (mod, ai) in enumerate(sorted(self.child.items())):
            ai = (ai.float_prec() % mod).long()
            res += ai * self.reconstruction_coeffs[i]
            N *= mod

        return FixedPrecisionTensor(
            field=N, base=self.base, precision_fractional=self.precision_fractional
        ).on(res % N)

    def compute_reconstruction_coeffs(self):
        """ Compute the coefficients needed for the reconstruction (see the detailed exaplanation in the
        docstring for the reconstruct method).
        """

        def modular_inverse(a, b):
            """ Computes the modular inverse x = a^(-1) mod b
            with Euclid's extended algorithm.
            The coefficient are computed and stored in a list in ascending order of moduli.
            The list is then turned to a tuple to be hashable and passed via the get_class_attributes method.
            """
            b0 = b
            x0, x1 = 0, 1
            if b == 1:
                return 1
            while a > 1:
                q = a // b
                a, b = b, a % b
                x0, x1 = x1 - q * x0, x0
            if x1 < 0:
                x1 += b0
            return x1

        N = 1
        for mod in self.child.keys():
            N *= mod

        coeffs = []
        for mod, ai in sorted(self.child.items()):
            yi = N // mod
            zi = modular_inverse(yi, mod)
            coeffs.append(yi * zi)

        return tuple(coeffs)

    @staticmethod
    @overloaded.module
    def torch(module):
        def add(self, other):
            return self.add(other)

        module.add = add

        def sub(self, other):
            return self.sub(other)

        module.sub = sub

        def mul(self, other):
            return self.mul(other)

        module.mul = mul

        def div(self, other):
            return self.div(other)

        module.div = div

    def share(self, *owners, field=None, crypto_provider=None):
        """ Share the tensor between several workers.
        This gives an AdditiveSharingTensor wrapped around the original CRT tensor
        """
        assert field is None or field == self.field, "field is chosen when fixing precision"
        for mod, res in self.child.items():
            ast_mod = res.share(*owners, field=mod, crypto_provider=None)
            self.child[mod] = ast_mod

        return self

    def get(self):
        """ Get back a tensor shared between several workers.
        """
        self_ = self.child
        for mod, res in self_.items():
            res.child = res.child.get().child
        return self.wrap()

    @staticmethod
    def simplify(tensor: "CRTTensor") -> tuple:
        """
        This function takes the attributes of a CRTTensor and saves them in a tuple
        Args:
            crttensor: a CRTTensor
        Returns:
            tuple: a tuple holding the unique attributes of the tensor
        """
        chain = None
        if hasattr(tensor, "child"):
            chain = syft.serde._simplify(tensor.child)

        return (
            tensor.id,
            tensor.base,
            tensor.precision_fractional,
            tensor.reconstruction_coeffs,
            chain,
        )

    @staticmethod
    def detail(worker, tensor_tuple: tuple) -> "CRTTensor":
        """
        This function reconstructs a CRTTensor given its attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the CRTTensor
        Returns:
            CRTTensor: a CRTTensor
        """
        tensor_id, tensor_base, tensor_precision_fractional, tensor_reconstruction_coeffs, chain = (
            tensor_tuple
        )

        tensor = syft.CRTTensor(
            base=tensor_base,
            precision_fractional=tensor_precision_fractional,
            # FIXME need to cast to tuple otherwise tensor_reconstruction_coeffs is an unashable list
            reconstruction_coeffs=tuple(tensor_reconstruction_coeffs),
            owner=worker,
            id=tensor_id,
        )

        if chain is not None:
            chain = syft.serde._detail(worker, chain)
            tensor.child = chain

        return tensor

    def get_class_attributes(self):
        return {
            "base": self.base,
            "precision_fractional": self.precision_fractional,
            # Solving coefficients can be reused in results of operations (see docstring of solve_system method)
            "reconstruction_coeffs": self.reconstruction_coeffs,
        }


_moduli_for_fields = {21: [3, 7]}
