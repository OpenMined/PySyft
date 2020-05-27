from collections import defaultdict
from enum import Enum

from syft.frameworks.torch.he.fv.util.numth import get_primes
from syft.frameworks.torch.he.fv.util.global_variable import DEFAULT_C0EFF_MODULUS_128
from syft.frameworks.torch.he.fv.util.global_variable import DEFAULT_C0EFF_MODULUS_192
from syft.frameworks.torch.he.fv.util.global_variable import DEFAULT_C0EFF_MODULUS_256


class SeqLevelType(Enum):
    """Represents standard security level according to the HomomorphicEncryption.org
    security standard. Can be used to select the suggested secure coefficient modulus.
    """

    TC128 = 128
    TC192 = 192
    TC256 = 256


class CoeffModulus:
    def bfv_default(self, poly_modulus_degree, seq_level=SeqLevelType.TC128):
        """Provide secure coefficient modulus for BFV scheme that guarantees
        a given security level when used with a given polynomial modulus,
        according to the HomomorphicEncryption.org security standard.

        Args:
            poly_modulus_degree: The value of the polynomial modulus.
            seq_level: (optional) The desired standard security level.

        Returns:
            A list of coefficient modulus.

        Raises;
            ValueError: if coefficient modulus are not available for provided
                polynomial modulus.

        """

        if seq_level == SeqLevelType.TC128:
            return DEFAULT_C0EFF_MODULUS_128[poly_modulus_degree]

        if seq_level == SeqLevelType.TC192:
            return DEFAULT_C0EFF_MODULUS_192[poly_modulus_degree]

        if seq_level == SeqLevelType.TC256:
            return DEFAULT_C0EFF_MODULUS_256[poly_modulus_degree]

        raise ValueError(f"{seq_level} is not a valid standard security level")

    def create(self, poly_modulus_degree, bit_sizes):
        """Generate coefficient modulus suitable for use with the specified
        polynomial modulus. The return value will be a list consisting of
        distinct prime numbers of bit-lengths as given in the bit_sizes parameter.

        Args:
            poly_modulus_degree: The value of the polynomial modulus.
            bit_sizes: (list) The bit-lengths of coefficient modulus value to be generated.

        Returns:
            A list of coefficient modulus.
        """

        count_table = defaultdict(lambda: 0)
        prime_table = defaultdict(lambda: 0)

        for size in bit_sizes:
            count_table[size] += 1

        for table_elt in count_table:
            prime_table[table_elt] = get_primes(
                poly_modulus_degree, table_elt, count_table[table_elt]
            )

        result = []
        for size in bit_sizes:
            result.append(prime_table[size][-1])
            prime_table[size].pop()

        return result
