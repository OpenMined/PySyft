from math import gcd

from syft.frameworks.torch.he.fv.util.operations import multiply_many_except
from syft.frameworks.torch.he.fv.util.operations import invert_mod


class RNSBase:
    """Generate RNSBase object with pre-computed values required in RNSTools class.

    Args:
        base: A list of Base values.

    Attributes:
        size: The number of base values given.
        base: A list of Base values.
        base_prod: An integer denoting the product of all base values.

        punctured_prod_list: A list of products of all base values except
        the base value at that index.

        inv_punctured_prod_mod_base_list: A list of values equal to modulus
        inverse of punctured_prod_list values.
    """

    def __init__(self, base):
        self.size = len(base)

        for i in range(self.size):
            if base[i] == 0:
                raise ValueError("rns_base is invalid")

            # The base must be coprime
            for j in base[:i]:
                if gcd(base[i], j) != 1:
                    raise ValueError("rns_base is invalid")

        self.base = base
        self.base_prod = None
        self.punctured_prod_list = [0] * self.size
        self.inv_punctured_prod_mod_base_list = [0] * self.size

        if self.size > 1:
            # Compute punctured product
            for i in range(self.size):
                self.punctured_prod_list[i] = multiply_many_except(self.base, self.size, i)

            # Compute the full product
            self.base_prod = self.punctured_prod_list[0] * self.base[0]

            # Compute inverses of punctured products mod primes
            for i in range(self.size):
                self.inv_punctured_prod_mod_base_list[i] = (
                    self.punctured_prod_list[i] % self.base[i]
                )
                self.inv_punctured_prod_mod_base_list[i] = invert_mod(
                    self.inv_punctured_prod_mod_base_list[i], self.base[i]
                )

        else:
            self.base_prod = self.base[0]
            self.punctured_prod_list[0] = 1
            self.inv_punctured_prod_mod_base_list[0] = 1
