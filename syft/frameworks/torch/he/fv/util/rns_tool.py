from syft.frameworks.torch.he.fv.util.operations import negate_mod
from syft.frameworks.torch.he.fv.util.operations import invert_mod
from syft.frameworks.torch.he.fv.util.operations import multiply_mod
from syft.frameworks.torch.he.fv.util.operations import poly_sub_mod
from syft.frameworks.torch.he.fv.util.base_converter import BaseConvertor
from syft.frameworks.torch.he.fv.util.rns_base import RNSBase
from syft.frameworks.torch.he.fv.util.numth import get_primes


class RNSTool:
    """A class to perform major operations required in the process of decryption
    in RNS variant of FV HE Scheme.

    Args:
        encryption_param (EncryptionParams): For extracting encryption parameters.
    """

    def __init__(self, encryption_param):
        self.coeff_count = encryption_param.poly_modulus
        self.q = encryption_param.coeff_modulus
        self.t = encryption_param.plain_modulus

        self.initialize()

    def initialize(self):
        base_q_size = len(self.q)
        # In some cases we might need to increase the size of the base B by one, namely we require
        # K*n*t*q^2<q*prod(B)*m_sk, where K takes into account cross terms when larger size
        # ciphertexts are used, and n is the "delta factor" for the ring. We reserve 32 bits
        # for K * n. Here the coeff modulus primes q_i are bounded to be
        total_coeff_bit_count = RNSBase(self.q).base_prod.bit_length()

        base_B_size = base_q_size
        if 32 + self.t.bit_length() + total_coeff_bit_count >= 61 * len(self.q) + 61:
            base_B_size += 1

        # Sample primes for B and two more primes: m_sk and gamma.
        baseconv_primes = get_primes(self.coeff_count, 50, base_B_size + 2)
        self.m_sk = baseconv_primes[0]
        self.gamma = baseconv_primes[1]
        base_B_primes = baseconv_primes[2:]

        self.prod_t_gamma_mod_q = [(self.t * self.gamma) % q for q in self.q]
        self.inv_gamma_mod_t = invert_mod(self.gamma, self.t)

        # Set m_tilde_ to a non-prime value
        self.m_tilde = 1 << 32

        # Populate the base arrays
        self.base_q = RNSBase(self.q)
        self.base_B = RNSBase(base_B_primes)
        self.base_Bsk = RNSBase(base_B_primes + [self.m_sk])
        self.base_Bsk_m_tilde = RNSBase(base_B_primes + [self.m_sk] + [self.m_tilde])

        if self.t != 0:
            self.base_t_gamma_size = 2
            self.base_t_gamma = RNSBase([self.t, self.gamma])

        # Set up BaseConvTool for q --> Bsk
        self.base_q_to_Bsk_conv = BaseConvertor(self.base_q, self.base_Bsk)

        # Set up BaseConvTool for q --> {m_tilde}
        self.base_q_to_m_tilde_conv = BaseConvertor(self.base_q, RNSBase([self.m_tilde]))

        # Set up BaseConvTool for B --> q
        self.base_B_to_q_conv = BaseConvertor(self.base_B, self.base_q)

        # Set up BaseConvTool for B --> {m_sk}
        self.base_B_to_m_sk_conv = BaseConvertor(self.base_B, RNSBase([self.m_sk]))

        if self.t != 0:
            # Base conversion: convert from q to {t, gamma}
            self.base_q_to_t_gamma_conv = BaseConvertor(self.base_q, self.base_t_gamma)

        # Compute prod(q)^(-1) mod m_tilde
        inv_prod_q_mod_m_tilde = self.base_q.base_prod % self.m_tilde
        self.inv_prod_q_mod_m_tilde = invert_mod(inv_prod_q_mod_m_tilde, self.m_tilde)

        # Compute m_tilde^(-1) mod Bsk
        self.inv_m_tilde_mod_Bsk = [0] * self.base_Bsk.size
        for i in range(self.base_Bsk.size):
            self.inv_m_tilde_mod_Bsk[i] = invert_mod(
                self.m_tilde % self.base_Bsk.base[i], self.base_Bsk.base[i]
            )

        # Compute prod(q) mod Bsk
        self.prod_q_mod_Bsk = [0] * self.base_Bsk.size
        for i in range(self.base_Bsk.size):
            self.prod_q_mod_Bsk[i] = self.base_q.base_prod % self.base_Bsk.base[i]

        # Compute prod(B)^(-1) mod m_sk
        self.inv_prod_B_mod_m_sk = self.base_B.base_prod % self.m_sk
        self.inv_prod_B_mod_m_sk = invert_mod(self.inv_prod_B_mod_m_sk, self.m_sk)

        # Compute prod(B) mod q
        self.prod_B_mod_q = [0] * self.base_q.size
        for i in range(self.base_q.size):
            self.prod_B_mod_q[i] = self.base_B.base_prod % self.base_q.base[i]

        # Compute prod(q)^(-1) mod Bsk
        self.inv_prod_q_mod_Bsk = [0] * self.base_Bsk.size
        for i in range(self.base_Bsk.size):
            self.inv_prod_q_mod_Bsk[i] = self.base_q.base_prod % self.base_Bsk.base[i]
            self.inv_prod_q_mod_Bsk[i] = invert_mod(
                self.inv_prod_q_mod_Bsk[i], self.base_Bsk.base[i]
            )

        if self.t != 0:
            # Compute -prod(q)^(-1) mod {t, gamma}
            self.neg_inv_q_mod_t_gamma = [0] * self.base_t_gamma_size
            for i in range(self.base_t_gamma_size):
                self.neg_inv_q_mod_t_gamma[i] = self.base_q.base_prod % self.base_t_gamma.base[i]
                self.neg_inv_q_mod_t_gamma[i] = invert_mod(
                    self.neg_inv_q_mod_t_gamma[i], self.base_t_gamma.base[i]
                )
                self.neg_inv_q_mod_t_gamma[i] = negate_mod(
                    self.neg_inv_q_mod_t_gamma[i], self.base_t_gamma.base[i]
                )

        # Compute q[last]^(-1) mod q[i] for i = 0..last-1
        # This is used by modulus switching and rescaling
        self.inv_q_last_mod_q = [0] * (base_q_size - 1)
        for i in range(base_q_size - 1):
            self.inv_q_last_mod_q[i] = invert_mod(self.q[-1], self.q[i])

    def divide_and_round_q_last_inplace(self, input):
        base_q_size = self.base_q.size
        last_ptr = input[base_q_size - 1]

        # Add (qi-1)/2 to change from flooring to rounding
        last_modulus = self.base_q.base[-1]
        half = last_modulus >> 1

        last_ptr = [(x + half) % last_modulus for x in last_ptr]

        temp_ptr = []
        for i in range(base_q_size - 1):
            temp_ptr = [x % self.base_q.base[i] for x in last_ptr]
            half_mod = half % self.base_q.base[i]

            temp_ptr = [(x - half_mod) % self.base_q.base[i] for x in temp_ptr]

            input[i] = poly_sub_mod(input[i], temp_ptr, self.base_q.base[i], self.coeff_count)

            # qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
            input[i] = [(x * self.inv_q_last_mod_q[i]) % self.base_q.base[i] for x in input[i]]

        return input

    def decrypt_scale_and_round(self, input):
        """Perform the remaining procedure of decryption process after getting the result of
        [c0 + c1 * sk + c2 * sk^2 ...]_q.

        Args:
            input: Result of [c0 + c1 * sk + c2 * sk^2 ...]_q.

        Returns:
            A 1-dim list representing plaintext polynomial of the decrypted result.
        """
        result = [0] * self.coeff_count

        # Computing |gamma * t|_qi * ct(s)
        temp = [
            [
                multiply_mod(input[j][i], self.prod_t_gamma_mod_q[j], self.base_q.base[j])
                for i in range(self.coeff_count)
            ]
            for j in range(len(self.q))
        ]

        temp_t_gamma = self.base_q_to_t_gamma_conv.fast_convert_list(temp, self.coeff_count)

        # Multiply by -prod(q)^(-1) mod {t, gamma}
        for j in range(self.base_t_gamma_size):
            for i in range(self.coeff_count):
                temp_t_gamma[j][i] = multiply_mod(
                    temp_t_gamma[j][i], self.neg_inv_q_mod_t_gamma[j], self.base_t_gamma.base[j]
                )

        # Need to correct values in temp_t_gamma (gamma component only) which are larger
        # than floor(gamma/2)
        gamma_div_2 = self.gamma >> 1

        # Now compute the subtraction to remove error and perform final multiplication by gamma
        # inverse mod t
        for i in range(self.coeff_count):
            # Need correction because of centered mod
            if temp_t_gamma[1][i] > gamma_div_2:

                # Compute -(gamma - a) instead of (a - gamma)
                result[i] = (
                    temp_t_gamma[0][i] + (self.gamma - temp_t_gamma[1][i]) % self.t
                ) % self.t
            else:
                # No correction needed
                result[i] = (temp_t_gamma[0][i] - temp_t_gamma[1][i]) % self.t

            # If this coefficient was non-zero, multiply by t^(-1)
            if 0 != result[i]:

                # Perform final multiplication by gamma inverse mod t
                result[i] = multiply_mod(result[i], self.inv_gamma_mod_t, self.t)

        return result

    def fastbconv_m_tilde(self, input):
        """
        Require: Input in q
        Ensure: Output in Bsk U {m_tilde}
        """

        # We need to multiply first the input with m_tilde mod q
        # This is to facilitate Montgomery reduction in the next step of multiplication
        # This is NOT an ideal approach: as mentioned in BEHZ16, multiplication by
        # m_tilde can be easily merge into the base conversion operation; however, then
        # we could not use the BaseConvertor as below without modifications.

        temp = [
            [
                multiply_mod(input[i][j], self.m_tilde, self.base_q.base[i])
                for j in range(self.coeff_count)
            ]
            for i in range(len(self.q))
        ]

        # Now convert to Bsk
        result = self.base_q_to_Bsk_conv.fast_convert_list(temp, self.coeff_count)

        # Finally convert to {m_tilde}
        result += self.base_q_to_m_tilde_conv.fast_convert_list(temp, self.coeff_count)
        return result

    def sm_mrq(self, input):
        """
        Require: Input in base Bsk U {m_tilde}
        Ensure: Output in base Bsk
        """
        m_tilde_div_2 = self.m_tilde >> 1
        result = []

        # Compute r_m_tilde; The last component of the input is mod m_tilde
        r_m_tilde = []
        for i in range(self.coeff_count):
            r_m_tilde.append(
                negate_mod(
                    multiply_mod(input[-1][i], self.inv_prod_q_mod_m_tilde, self.m_tilde),
                    self.m_tilde,
                )
            )

        for k in range(self.base_Bsk.size):
            base_Bsk_elt = self.base_Bsk.base[k]
            inv_m_tilde_mod_Bsk_elt = self.inv_m_tilde_mod_Bsk[k]
            prod_q_mod_Bsk_elt = self.prod_q_mod_Bsk[k]

            temp_list = []
            for i in range(self.coeff_count):
                # We need centered reduction of r_m_tilde modulo Bsk. Note that m_tilde is chosen
                # to be a power of two so we have '>=' below.
                temp = r_m_tilde[i]
                if temp >= m_tilde_div_2:
                    temp += base_Bsk_elt - self.m_tilde

                # Compute (input + q*r_m_tilde)*m_tilde^(-1) mod Bsk
                temp_list.append(
                    (
                        ((prod_q_mod_Bsk_elt * temp + input[k][i]) % base_Bsk_elt)
                        * inv_m_tilde_mod_Bsk_elt
                    )
                    % base_Bsk_elt
                )

            result.append(temp_list)
        return result

    def fast_floor(self, input):
        """
        Require: Input in base q U Bsk
        Ensure: Output in base Bsk
        """
        base_q_size = self.base_q.size
        base_Bsk_size = self.base_Bsk.size

        # Convert q -> Bsk
        result = self.base_q_to_Bsk_conv.fast_convert_list(input[:base_q_size], self.coeff_count)

        for i in range(base_Bsk_size):
            base_Bsk_elt = self.base_Bsk.base[i]
            inv_prod_q_mod_Bsk_elt = self.inv_prod_q_mod_Bsk[i]

            for k in range(self.coeff_count):
                result[i][k] = (
                    (input[i + base_q_size][k] + (base_Bsk_elt - result[i][k]))
                    * inv_prod_q_mod_Bsk_elt
                ) % base_Bsk_elt

        return result

    def fastbconv_sk(self, input):
        """
        Require: Input in base Bsk
        Ensure: Output in base q
        """

        # Fast convert B -> q; input is in Bsk but we only use B
        result = self.base_B_to_q_conv.fast_convert_list(input[:-1], self.coeff_count)

        # Compute alpha_sk
        # Fast convert B -> {m_sk}; input is in Bsk but we only use B
        temp = self.base_B_to_m_sk_conv.fast_convert_list(input[:-1], self.coeff_count)

        # Take the m_sk part of input, subtract from temp, and multiply by inv_prod_B_mod_m_sk_
        # input_sk is allocated in input + (base_B_size * coeff_count_)
        alpha_sk_ptr = []
        for i in range(self.coeff_count):
            # It is not necessary for the negation to be reduced modulo the small prime
            alpha_sk_ptr.append(
                ((temp[0][i] + (self.m_sk - input[-1][i])) * self.inv_prod_B_mod_m_sk) % self.m_sk
            )

        # alpha_sk is now ready for the Shenoy-Kumaresan conversion; however, note that our
        # alpha_sk here is not a centered reduction, so we need to apply a correction below.
        m_sk_div_2 = self.m_sk >> 1
        for i in range(self.base_q.size):
            base_q_elt = self.base_q.base[i]
            prod_B_mod_q_elt = self.prod_B_mod_q[i]
            for k in range(self.coeff_count):
                # Correcting alpha_sk since it represents a negative value
                if alpha_sk_ptr[k] > m_sk_div_2:
                    result[i][k] = (
                        (prod_B_mod_q_elt * (self.m_sk - alpha_sk_ptr[k])) + result[i][k]
                    ) % base_q_elt
                else:
                    result[i][k] = (
                        ((base_q_elt - self.prod_B_mod_q[i]) * alpha_sk_ptr[k]) + result[i][k]
                    ) % base_q_elt
        return result
