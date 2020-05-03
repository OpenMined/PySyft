from syft.frameworks.torch.he.fv.util.global_variable import gamma
from syft.frameworks.torch.he.fv.util.operations import negate_int_mod
from syft.frameworks.torch.he.fv.util.operations import try_invert_int_mod
from syft.frameworks.torch.he.fv.util.base_converter import BaseConvertor
from syft.frameworks.torch.he.fv.util.rns_base import RNSBase


class RNSTool:
    def __init__(self, poly_modulus_degree, q, t):
        self._coeff_count = poly_modulus_degree
        self.base_q = RNSBase(q)
        self.base_q_size = len(q)
        self._t = t
        self.base_t_gamma = RNSBase([t, gamma])
        self.base_t_gamma_size = 2
        self.prod_t_gamma_mod_q = [(self._t * gamma) % q for q in self.base_q.base]
        self._inv_gamma_mod_t = try_invert_int_mod(gamma, self._t)
        self.neg_inv_q_mod_t_gamma = [0] * self.base_t_gamma_size

        for i in range(self.base_t_gamma_size):
            self.neg_inv_q_mod_t_gamma[i] = self.base_q.base_prod % self.base_t_gamma.base[i]
            self.neg_inv_q_mod_t_gamma[i] = negate_int_mod(
                self.neg_inv_q_mod_t_gamma[i], self.base_t_gamma.base[i]
            )

    def decrypt_scale_and_round(self, input):
        result = [0] * self._coeff_count

        # Computing |gamma * t|_qi * ct(s)
        temp = [0] * self._coeff_count * self.base_q_size

        for j in range(self.base_q_size):
            for i in range(self._coeff_count):
                temp[i + j * self._coeff_count] = (
                    input[i + j * self._coeff_count] * self.prod_t_gamma_mod_q[j]
                ) % self.base_q.base[j]

        # Convert from q to {t, gamma}
        base_q_to_t_gamma_conv = BaseConvertor(self.base_q, self.base_t_gamma)
        temp_t_gamma = base_q_to_t_gamma_conv.fast_convert_array(temp, self._coeff_count)

        # Multiply by -prod(q)^(-1) mod {t, gamma}
        for j in range(self.base_t_gamma_size):
            for i in range(self._coeff_count):
                temp_t_gamma[i + j * self._coeff_count] = (
                    temp_t_gamma[i + j * self._coeff_count] * self.neg_inv_q_mod_t_gamma[j]
                ) % self.base_t_gamma.base[j]

        # Need to correct values in temp_t_gamma (gamma component only) which are larger than floor(gamma/2)
        gamma_div_2 = self.base_t_gamma.base[1] >> 1

        # Now compute the subtraction to remove error and perform final multiplication by gamma inverse mod t
        for i in range(self._coeff_count):
            # Need correction because of centered mod
            if temp_t_gamma[i + self._coeff_count] > gamma_div_2:

                # Compute -(gamma - a) instead of (a - gamma)
                result[i] = (
                    temp_t_gamma[i] + (gamma - temp_t_gamma[i + self._coeff_count] % self._t)
                ) % self._t
            else:
                # No correction needed
                result[i] = (temp_t_gamma[i] + temp_t_gamma[i + self._coeff_count]) % self._t

            # If this coefficient was non-zero, multiply by t^(-1)
            if 0 != result[i]:

                # Perform final multiplication by gamma inverse mod t
                result[i] = (result[i] * self._inv_gamma_mod_t) % self._t

        return result
