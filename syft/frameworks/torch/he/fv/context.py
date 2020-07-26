import math
from functools import reduce

# from syft.frameworks.torch.he.fv.modulus import CoeffModulus
# from syft.frameworks.torch.he.fv.util.operations import get_significant_count
from syft.frameworks.torch.he.fv.util.rns_tool import RNSTool
from syft.frameworks.torch.he.fv.util.global_variable import COEFF_MOD_COUNT_MAX
from syft.frameworks.torch.he.fv.util.global_variable import COEFF_MOD_COUNT_MIN
from syft.frameworks.torch.he.fv.util.global_variable import COEFF_MOD_BIT_COUNT_MAX
from syft.frameworks.torch.he.fv.util.global_variable import COEFF_MOD_BIT_COUNT_MIN
from syft.frameworks.torch.he.fv.util.global_variable import POLY_MOD_DEGREE_MAX
from syft.frameworks.torch.he.fv.util.global_variable import POLY_MOD_DEGREE_MIN


class Context:
    """A class used as for holding and easily supplying of all the general
    parameters required throughout the implementation.

    Attributes:
        param: An EncryptionParams object.
        coeff_div_plain_modulus: A list of float values equal to (q[i]/t),
            In research papers denoted by delta.
        rns_tool: A RNSTool class instance.
    """

    def __init__(self, params):

        # Validation of params provided for the encryption scheme.
        self.validate(params)

        self.param = params

        self.rns_tool = RNSTool(params)

    def validate(self, params):

        # The number of coeff moduli is restricted to 62
        if (
            len(params.coeff_modulus) > COEFF_MOD_COUNT_MAX
            or len(params.coeff_modulus) < COEFF_MOD_COUNT_MIN
        ):
            raise RuntimeError(
                f"Invalid coefficient modulus count {len(params.coeff_modulus)}, "
                + "should be in range [1, 62]"
            )

        # Check for the range of coefficient modulus primes.
        for i in range(len(params.coeff_modulus)):
            if params.coeff_modulus[i] >> COEFF_MOD_BIT_COUNT_MAX or not (
                params.coeff_modulus[i] >> (COEFF_MOD_BIT_COUNT_MIN - 1)
            ):
                raise RuntimeError(
                    f"Invalid coefficient modulus values {params.coeff_modulus[i]}, "
                    + "should be in smaller than 60 bit number."
                )

            # Check for the relative prime of coefficient modulus primes
            for j in range(i):
                if math.gcd(params.coeff_modulus[i], params.coeff_modulus[j]) > 1:
                    raise RuntimeError("Coefficient modulus are not pairwise relatively prime")

        # Compute the product of all coeff moduli
        total_coeff_modulus = reduce(lambda x, y: x * y, params.coeff_modulus)

        # Check polynomial modulus degree and create poly_modulus
        poly_modulus = params.poly_modulus
        coeff_count_power = poly_modulus & (poly_modulus - 1)
        if (
            poly_modulus < POLY_MOD_DEGREE_MIN
            or poly_modulus > POLY_MOD_DEGREE_MAX
            or coeff_count_power < 0
        ):
            raise RuntimeError("Invalid polynomial modulus value")

        # TODO: Check if the parameters are secure according to HomomorphicEncryption.org standards
        # total_coeff_mod_bit_count = get_significant_count(total_coeff_modulus)
        # if total_coeff_mod_bit_count > CoeffModulus.max_bit_count(poly_modulus, self.sec_level):
        #     raise RuntimeError(
        #         "Coefficient modulus is not appropriate for the specified security level."
        #     )

        if params.plain_modulus >> COEFF_MOD_BIT_COUNT_MAX or not (
            params.plain_modulus >> (COEFF_MOD_COUNT_MIN - 1)
        ):
            raise RuntimeError("Invalid plain modulus values")

        if params.plain_modulus > total_coeff_modulus:
            raise RuntimeError("Plain modulus cannot be greater than total coefficient modulus")

        # Check all coeff moduli are relatively prime to plain_modulus
        for i in range(len(params.coeff_modulus)):
            if math.gcd(params.coeff_modulus[i], params.plain_modulus) > 1:
                raise RuntimeError(
                    "Coefficient modulus are not relatively prime with plain modulus"
                )

        # A list containing values of coeff_mod[i] / plain_mod for one time computation
        # and useful in encryption process.
        self.coeff_div_plain_modulus = [x / params.plain_modulus for x in params.coeff_modulus]
