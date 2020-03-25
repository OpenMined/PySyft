from syft.frameworks.torch.he.fv.util.get_primes import get_primes
from syft.frameworks.torch.he.fv.fv_std_param import *
from syft.frameworks.torch.he.fv.util.global_variable import *


class seq_level_type:
    """ Represents a standard security level according to the HomomorphicEncryption.org
    security standard. The value sec_level_type(none) signals that no standard
    security level should be imposed."""

    TC128 = 128
    TC192 = 192
    TC256 = 256


class CoeffModulus:
    def MaxBitCount(self, poly_modulus_degree, seq_level=seq_level_type.tc128):
        """Returns the largest bit-length of the coefficient modulus, i.e., bit-length
        of the product of the primes in the coefficient modulus, that guarantees
        a given security level when using a given poly_modulus_degree, according
        to the HomomorphicEncryption.org security standard.

        Args:
            poly_modulus_degree: The value of the poly_modulus_degree
        encryption parameter
            seq_level: (optional) The desired standard security level

        Returns:
            A integer denoting the largest allowed bit counts for coeff_modulus.

        Raises:
            ValueError: seq_level does not match with any standard security level.
        """

        if seq_level == seq_level_type.TC128:
            return FV_STD_PARMS_128_TC(poly_modulus_degree)

        if seq_level == seq_level_type.TC192:
            return FV_STD_PARMS_192_TC(poly_modulus_degree)

        if seq_level == seq_level_type.TC256:
            return FV_STD_PARMS_256_TC(poly_modulus_degree)

        raise ValueError(f"{seq_level} is not a valid standard security level")

    def BFVDefault(self, poly_modulus_degree, seq_level=seq_level_type.tc128):
        """Returns a default coefficient modulus for the BFV scheme that guarantees
        a given security level when using a given poly_modulus_degree, according
        to the HomomorphicEncryption.org security standard. Note that all security
        guarantees are lost if the output is used with encryption parameters with
        a mismatching value for the poly_modulus_degree.

        Args:
            poly_modulus_degree: The value of the poly_modulus_degree
        encryption parameter
            seq_level: (optional) The desired standard security level
        """

        if seq_level == seq_level_type.TC128:
            return DEFAULT_C0EFF_MODULUS_128[poly_modulus_degree]

        if seq_level == seq_level_type.TC192:
            return DEFAULT_C0EFF_MODULUS_192[poly_modulus_degree]

        if seq_level == seq_level_type.TC256:
            return DEFAULT_C0EFF_MODULUS_256[poly_modulus_degree]

        raise ValueError(f"{seq_level} is not a valid standard security level")

    def Create(self, poly_modulus_degree, bit_sizes):
        """Returns a custom coefficient modulus suitable for use with the specified
        poly_modulus_degree. The return value will be a list consisting of
        distinct prime numbers of bit-lengths as given in the bit_sizes parameter.

        Args:
            poly_modulus_degree: The value of the poly_modulus_degree encryption parameter
            bit_sizes: (list) The bit-lengths of the primes to be generated
        """

        count_table = {}
        prime_table = {}

        for size in bit_sizes:
            count_table[size] += 1

        for table_elt in count_table:
            prime_table[table_elt[0]] = get_primes(poly_modulus_degree, table_elt[0], table_elt[1])

        result = []
        for size in bit_sizes:
            result.append(prime_table[size][-1])
            prime_table[size].pop()

        return result


class PlainModulus:
    """This class contains static methods for creating a plaintext modulus easily."""

    def Batching(self, poly_modulus_degree, bit_size):
        """Creates a prime number for use as plain_modulus encryption
        parameter that supports batching with a given poly_modulus_degree.

        Args:
            poly_modulus_degree: The value of the poly_modulus_degree
        encryption parameter
            bit_size: The bit-length of the prime to be generated
        """
        return CoeffModulus.Create(poly_modulus_degree, bit_size[0])

    def Batching(self, poly_modulus_degree, bit_sizes):
        """Creates several prime number that can be used as plain_modulus encryption parameters,
         each supporting batching with a given poly_modulus_degree

        Args:
            poly_modulus_degree: The value of the poly_modulus_degree
        encryption parameter
            bit_sizes: (list) The bit-lengths of the primes to be generated
        """
        return CoeffModulus.Create(poly_modulus_degree, bit_sizes)
