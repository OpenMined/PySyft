from math import gcd

from syft.frameworks.torch.he.fv.util.operations import multiply_many_except
from syft.frameworks.torch.he.fv.util.operations import invert_mod


class RNSBase:
    """A model class used for creating basic blocks required in RNSTools class with pre-computed attributes.

    Attributes:
        size: The number of base values given.
        base: A list of Base values.
        base_prod: An integer denoting the product of all base values.
        punctured_prod_list: A list of products of all base values except the base value at that index.
        inv_punctured_prod_mod_base_list: A list of values equal to modulus inverse of punctured_prod_list values.
    """

    def __init__(self, base):
        self._size = len(base)

        for i in range(self._size):
            if base[i] == 0:
                raise ValueError("rns_base is invalid")

            # The base must be coprime
            for j in base[:i]:
                if gcd(base[i], j) != 1:
                    raise ValueError("rns_base is invalid")

        self._base = base
        self._base_prod = None
        self._punctured_prod_list = [0] * self._size
        self._inv_punctured_prod_mod_base_list = [0] * self._size

        if self._size > 1:
            # Compute punctured product
            for i in range(self._size):
                self._punctured_prod_list[i] = multiply_many_except(self._base, self._size, i)

            # Compute the full product
            self._base_prod = self._punctured_prod_list[0] * self._base[0]

            # Compute inverses of punctured products mod primes
            for i in range(self._size):
                self._inv_punctured_prod_mod_base_list[i] = (
                    self._punctured_prod_list[i] % self._base[i]
                )
                self._inv_punctured_prod_mod_base_list[i] = invert_mod(
                    self._inv_punctured_prod_mod_base_list[i], self._base[i]
                )

        else:
            self._base_prod = self._base[0]
            self._punctured_prod_list[0] = 1
            self._inv_punctured_prod_mod_base_list[0] = 1

    @property
    def base(self):
        return self._base

    @property
    def size(self):
        return self._size

    @property
    def base_prod(self):
        return self._base_prod

    @property
    def punctured_prod_list(self):
        return self._punctured_prod_list

    @property
    def inv_punctured_prod_mod_base_list(self):
        return self._inv_punctured_prod_mod_base_list
