import copy
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
    def __init__(self, params):
        """A class used as for holding and supplying all the general
        parameters required throughout the implementation.

        Args:
            An EncryptionParams object with polynomial modulus, coefficient modulus and
            plain modulus set.
        """
        # Validation of params provided for the encryption scheme.
        self.key_param_id = params.param_id
        self.context_data_map = {}
        self.context_data_map[self.key_param_id] = ContextData(params)

        if len(params.coeff_modulus) > 1:
            self.first_param_id = self.create_next_context_data(self.key_param_id)
        else:
            self.first_param_id = self.key_param_id
        self.last_param_id = self.first_param_id

        prev_parms_id = self.first_param_id
        while len(self.context_data_map[prev_parms_id].param.coeff_modulus) > 1:
            prev_parms_id = self.create_next_context_data(prev_parms_id)
            self.last_param_id = prev_parms_id

    def create_next_context_data(self, prev_parms_id):
        """Generate next context data in context chain by dropping the last coefficient modulus.

        Args:
            prev_params_id: Id of previous context data in context chain.
        Returns:
            Id of newly generated context data in context chain.
        """
        next_parms = copy.deepcopy(self.context_data_map[prev_parms_id].param)
        next_coeff_modulus = next_parms.coeff_modulus[:-1]
        next_parms.set_coeff_modulus(next_coeff_modulus)
        next_param_id = next_parms.param_id

        assert next_param_id != prev_parms_id

        next_context_data = ContextData(next_parms)
        self.context_data_map[next_param_id] = next_context_data
        self.context_data_map[prev_parms_id].next_context_id = next_param_id
        self.context_data_map[next_param_id].prev_context_id = prev_parms_id
        return next_param_id


class ContextData:
    def __init__(self, param, prev_context_id=None, next_context_id=None):
        """A class used to hold parameters and perform parameters validation.

        Args:
            param: An EncryptionParams object with polynomial modulus, coefficient modulus and
            plain modulus set.
            prev_context_id: Id of the previous ContextData object in context chain.
            next_context_id: Id of the next ContextData object in context chain.

        Attributes:
            prev_context_id: Id of the previous ContextData object in context chain.
            next_context_id: Id of the next ContextData object in context chain.
            coeff_div_plain_modulus: A list of float values equal to (q[i]/t),
                In research papers denoted by delta.
            rns_tool: A RNSTool class instance.
        """
        self.validate(param)
        self.param = param
        self.prev_context_id = prev_context_id
        self.next_context_id = next_context_id

        # A list containing values of coeff_mod[i] / plain_mod for one time computation
        # and useful in encryption process.
        self.coeff_div_plain_modulus = [x / param.plain_modulus for x in param.coeff_modulus]

        self.rns_tool = RNSTool(param)

    def validate(self, params):
        """Performs parameter validation on the provided encryption parameters.

        Args:
            params: An EncryptionParams object with polynomial modulus, coefficient modulus and
            plain modulus set.
        """
        # The number of coeff moduli is restricted to 62
        if (
            len(params.coeff_modulus) > COEFF_MOD_COUNT_MAX
            or len(params.coeff_modulus) < COEFF_MOD_COUNT_MIN
        ):
            raise RuntimeError(
                f"Invalid coefficient modulus count {len(params.coeff_modulus)}, "
                + "should be in range [{COEFF_MOD_COUNT_MIN}, {COEFF_MOD_COUNT_MAX}]"
            )

        # Check for the range of coefficient modulus primes.
        for i in range(len(params.coeff_modulus)):
            if params.coeff_modulus[i] >> COEFF_MOD_BIT_COUNT_MAX or not (
                params.coeff_modulus[i] >> (COEFF_MOD_BIT_COUNT_MIN - 1)
            ):
                raise RuntimeError(
                    f"Invalid coefficient modulus values {params.coeff_modulus[i]}, "
                    + "should be in smaller than {COEFF_MOD_BIT_COUNT_MAX} bit number."
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
