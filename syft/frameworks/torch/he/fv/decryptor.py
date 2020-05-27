from numpy.polynomial import polynomial as poly


from syft.frameworks.torch.he.fv.plaintext import PlainText
from syft.frameworks.torch.he.fv.util.operations import get_significant_count
from syft.frameworks.torch.he.fv.util.operations import poly_add_mod
from syft.frameworks.torch.he.fv.util.operations import poly_mul_mod


class Decryptor:
    """Decrypts Ciphertext objects into Plaintext objects.

    Args:
        context (Context): Context for extracting encryption parameters.
        secret_key: A secret key from same pair of keys(secretkey or publickey) used in encryptor.
    """

    def __init__(self, context, secret_key):
        self._context = context
        self._coeff_modulus = context.param.coeff_modulus
        self._coeff_count = context.param.poly_modulus
        self._secret_key = secret_key.data

    def decrypt(self, encrypted):
        """Decrypts the encrypted ciphertext objects.

        Args:
            encrypted: A ciphertext object which has to be decrypted.

        Returns:
            A PlainText object containing the decrypted result.
        """

        # Calculate [c0 + c1 * sk + c2 * sk^2 ...]_q
        temp_product_modq = self._mul_ct_sk(encrypted.data)

        # Divide scaling variant using BEHZ FullRNS techniques
        result = self._context.rns_tool.decrypt_scale_and_round(temp_product_modq)

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
        phase = encrypted[0]

        secret_key_array = self._get_sufficient_sk_power(len(encrypted))

        for j in range(1, len(encrypted)):
            for i in range(len(self._coeff_modulus)):
                phase[i] = poly_add_mod(
                    poly_mul_mod(
                        encrypted[j][i], secret_key_array[j - 1][i], self._coeff_modulus[i]
                    ),
                    phase[i],
                    self._coeff_modulus[i],
                )

        return phase

    def _get_sufficient_sk_power(self, max_power):
        """Generate an list of secret key polynomial raised to 1...max_power.

        Args:
            max_power: heighest power up to which we want to raise secretkey.

        Returns:
            A 2-dim list having secretkey powers.
        """
        sk_power = [[] for _ in range(max_power)]

        sk_power[0] = self._secret_key

        for i in range(2, max_power + 1):
            for j in range(len(self._coeff_modulus)):
                sk_power[i - 1].append(poly.polypow(self._secret_key[j], i).astype(int).tolist())
        return sk_power
