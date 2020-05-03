from math import gcd

from syft.frameworks.torch.he.fv.util.operations import multiply_many_except


class RNSBase:
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
        self._punctured_prod_array = [0] * self._size
        self._inv_punctured_prod_mod_base_array = [0] * self._size

        if self._size > 1:
            # Compute punctured product
            for i in range(self._size):
                self._punctured_prod_array[i] = multiply_many_except(self._base, self._size, i)

            # Compute the full product
            self._base_prod = self._punctured_prod_array[0] * self._base[0]

            # Compute inverses of punctured products mod primes
            for i in range(self._size):
                self._inv_punctured_prod_mod_base_array[i] = (
                    self._punctured_prod_array[i] % self._base[i]
                )

        else:
            self._base_prod = self._base[0]
            self._punctured_prod_array = [1]
            self._inv_punctured_prod_mod_base_array = [1]

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
    def punctured_prod_array(self):
        return self._punctured_prod_array

    @property
    def inv_punctured_prod_mod_base_array(self):
        return self._inv_punctured_prod_mod_base_array
