from math import gcd

from syft.frameworks.torch.he.fv.util.global_variable import gamma
from syft.frameworks.torch.he.fv.util.operations import multiply_many_except


class RNSBase:
    def __init__(self, base):
        size = len(base)

        for i in base:
            if i == 0:
                raise ValueError("rns_base is invalid")

            # The base must be coprime
            for j in base[:i]:
                if gcd(i, j) != 1:
                    raise ("rns_base is invalid")

        self._base = base
        self._base_prod = [0] * size
        self._punctured_prod_array = [0] * size
        self._inv_punctured_prod_mod_base_array = [0] * size

        if size > 1:
            # Compute punctured product
            for i in range(size):
                self._punctured_prod_array[i] = multiply_many_except(self._base, size, i)

            # Compute the full product
            self._base_prod = self._punctured_prod_array
            self._base_prod[0] = self._base_prod[0] * self._base[0]

            # Compute inverses of punctured products mod primes
            for i in range(size):
                self._inv_punctured_prod_mod_base_array[i] = (
                    self._punctured_prod_array[i] % self._base[i]
                )

        else:
            self._base_prod_ = base
            self._punctured_prod_array = 1
            self._inv_punctured_prod_mod_base_array_ = 1
