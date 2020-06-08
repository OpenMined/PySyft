from syft.frameworks.torch.he.fv.util.global_variable import gamma
from syft.frameworks.torch.he.fv.util.operations import negate_mod
from syft.frameworks.torch.he.fv.util.operations import invert_mod
from syft.frameworks.torch.he.fv.util.operations import multiply_mod
from syft.frameworks.torch.he.fv.util.base_converter import BaseConvertor
from syft.frameworks.torch.he.fv.util.rns_base import RNSBase


class RNSTool:
    """A class performing major operations required in the process of decryption
    in RNS variant of FV HE Scheme.

    After the multiplication of secret key with the ciphertext as [ct0 + ct1 * sk + ct2 * sk^2...]
    we apply the decrypt_scale_and_round method of this class to get the plaintext object.

    Args:
        encryption_param (EncryptionParams): For extracting encryption parameters.
    """

    def __init__(self, encryption_param):
        n = encryption_param.poly_modulus
        q = encryption_param.coeff_modulus
        t = encryption_param.plain_modulus

        self._coeff_count = n
        self.base_q = RNSBase(q)
        self.base_q_size = len(q)
        self._t = t
        self._base_t_gamma = RNSBase([t, gamma])
        self._base_t_gamma_size = 2
        self.prod_t_gamma_mod_q = [(t * gamma) % q for q in self.base_q.base]
        self._inv_gamma_mod_t = invert_mod(gamma, self._t)

        # Compute -prod(q)^(-1) mod {t, gamma}
        self.neg_inv_q_mod_t_gamma = [0] * self._base_t_gamma_size
        for i in range(self._base_t_gamma_size):
            self.neg_inv_q_mod_t_gamma[i] = self.base_q.base_prod % self._base_t_gamma.base[i]
            self.neg_inv_q_mod_t_gamma[i] = invert_mod(
                self.neg_inv_q_mod_t_gamma[i], self._base_t_gamma.base[i]
            )
            self.neg_inv_q_mod_t_gamma[i] = negate_mod(
                self.neg_inv_q_mod_t_gamma[i], self._base_t_gamma.base[i]
            )

    def decrypt_scale_and_round(self, input):
        """Perform the remaining procedure of decryptions process after getting the result of
        [c0 + c1 * sk + c2 * sk^2 ...]_q.

        Args:
            input: Result of [c0 + c1 * sk + c2 * sk^2 ...]_q.

        Returns:
            A 1-dim list representing plaintext polynomial of the decrypted result.
        """
        result = [0] * self._coeff_count

        # Computing |gamma * t|_qi * ct(s)
        temp = [0] * self.base_q_size
        for i in range(self.base_q_size):
            temp[i] = [0] * self._coeff_count

        for j in range(self.base_q_size):
            for i in range(self._coeff_count):
                temp[j][i] = multiply_mod(
                    input[j][i], self.prod_t_gamma_mod_q[j], self.base_q.base[j]
                )

        # Base conversion: convert from q to {t, gamma}
        base_q_to_t_gamma_conv = BaseConvertor(self.base_q, self._base_t_gamma)
        temp_t_gamma = base_q_to_t_gamma_conv.fast_convert_list(temp, self._coeff_count)

        # Multiply by -prod(q)^(-1) mod {t, gamma}
        for j in range(self._base_t_gamma_size):
            for i in range(self._coeff_count):
                temp_t_gamma[j][i] = multiply_mod(
                    temp_t_gamma[j][i], self.neg_inv_q_mod_t_gamma[j], self._base_t_gamma.base[j]
                )

        # Need to correct values in temp_t_gamma (gamma component only) which are larger
        # than floor(gamma/2)
        gamma_div_2 = gamma >> 1

        # Now compute the subtraction to remove error and perform final multiplication by gamma
        # inverse mod t
        for i in range(self._coeff_count):
            # Need correction because of centered mod
            if temp_t_gamma[1][i] > gamma_div_2:

                # Compute -(gamma - a) instead of (a - gamma)
                result[i] = (temp_t_gamma[0][i] + (gamma - temp_t_gamma[1][i]) % self._t) % self._t
            else:
                # No correction needed
                result[i] = (temp_t_gamma[0][i] - temp_t_gamma[1][i]) % self._t

            # If this coefficient was non-zero, multiply by t^(-1)
            if 0 != result[i]:

                # Perform final multiplication by gamma inverse mod t
                result[i] = multiply_mod(result[i], self._inv_gamma_mod_t, self._t)

        return result
