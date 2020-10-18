import copy

from syft.frameworks.torch.he.fv.plaintext import PlainText
from syft.frameworks.torch.he.fv.util.operations import get_significant_count
from syft.frameworks.torch.he.fv.util.operations import poly_add_mod
from syft.frameworks.torch.he.fv.util.operations import poly_mul_mod


class Decryptor:
    """Decrypts Ciphertext objects into Plaintext objects.

    Args:
        context (Context): Context for extracting encryption parameters.
        secret_key_list: A list to store secret key powers.
    """

    def __init__(self, context, secret_key):
        self.context = context
        self.secret_key_list = [secret_key.data]

    def decrypt(self, encrypted):
        """Decrypts the encrypted ciphertext objects.

        Args:
            encrypted: A ciphertext object which has to be decrypted.

        Returns:
            A PlainText object containing the decrypted result.
        """
        context_data = self.context.context_data_map[encrypted.param_id]

        # Calculate [c0 + c1 * sk + c2 * sk^2 ...]_q
        temp_product_modq = self._mul_ct_sk(copy.deepcopy(encrypted))

        # Divide scaling variant using BEHZ FullRNS techniques
        result = context_data.rns_tool.decrypt_scale_and_round(temp_product_modq)

        # removing leading zeroes in plaintext representation.
        plain_coeff_count = get_significant_count(result)
        return PlainText(result[:plain_coeff_count])

    def _mul_ct_sk(self, encrypted):
        """Calculate [c0 + c1 * sk + c2 * sk^2 ...]_q

        where [c0, c1, ...] represents ciphertext element and sk^n represents
        secret key raised to the power n.

        Args:
            encrypted: A ciphertext object of encrypted data.

        Returns:
            A 2-dim list containing result of [c0 + c1 * sk + c2 * sk^2 ...]_q.
        """
        context_data = self.context.context_data_map[encrypted.param_id]
        coeff_modulus = context_data.param.coeff_modulus
        coeff_count = context_data.param.poly_modulus
        encrypted = encrypted.data
        phase = encrypted[0]

        secret_key_list = self._get_sufficient_sk_power(len(encrypted) - 1)

        for j in range(1, len(encrypted)):
            for i in range(len(coeff_modulus)):
                phase[i] = poly_add_mod(
                    poly_mul_mod(
                        encrypted[j][i],
                        secret_key_list[j - 1][i],
                        coeff_modulus[i],
                        coeff_count,
                    ),
                    phase[i],
                    coeff_modulus[i],
                    coeff_count,
                )
        return phase

    def _get_sufficient_sk_power(self, max_power):
        """Generate an list of secret key polynomial raised to 1...max_power.

        Args:
            max_power: heighest power up to which we want to raise secretkey.

        Returns:
            A 2-dim list having secretkey powers.
        """
        param = self.context.context_data_map[self.context.key_param_id].param
        coeff_modulus = param.coeff_modulus
        coeff_count = param.poly_modulus
        if max_power == len(self.secret_key_list):
            return self.secret_key_list

        while len(self.secret_key_list) < max_power:
            sk_extra_power = [0] * len(coeff_modulus)
            for i in range(len(coeff_modulus)):
                sk_extra_power[i] = poly_mul_mod(
                    self.secret_key_list[-1][i],
                    self.secret_key_list[0][i],
                    coeff_modulus[i],
                    coeff_count,
                )
            self.secret_key_list.append(sk_extra_power)

        return self.secret_key_list
