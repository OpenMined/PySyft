from numpy.polynomial import polynomial as poly


from syft.frameworks.torch.he.fv.plaintext import PlainText
from syft.frameworks.torch.he.fv.util.operations import get_significant_count
from syft.frameworks.torch.he.fv.util.operations import poly_mul
from syft.frameworks.torch.he.fv.util.operations import poly_add


class Decryptor:
    """Decrypts Ciphertext objects into Plaintext objects. Constructing a Decryptor
    requires a Context object with valid encryption parameters, and the secret key.
    """

    def __init__(self, context, secret_key):
        self._context = context
        self._coeff_modulus = context.param.coeff_modulus
        self._coeff_count = context.param.poly_modulus_degree
        self._secret_key = secret_key.data

    def decrypt(self, encrypted):
        """Decrypts the encrypted ciphertext objects and return plaintext object

        Args:
            encrypted: A ciphertext object which has to be decrypted.
        """

        # Calculate [c0 + c1 * sk]_q
        temp_product_modq = self.mul_ct_sk(encrypted.data)

        # Divide scaling variant using BEHZ FullRNS techniques
        result = self._context.rns_tool.decrypt_scale_and_round(temp_product_modq)

        # removing leading zeroes in plaintext representation.
        plain_coeff_count = get_significant_count(result)
        return PlainText(result[:plain_coeff_count])

    def mul_ct_sk(self, encrypted):
        """calculate and return the value of [c0 + c1 * sk]_q
        where [c0, c1] denotes encrypted ciphertext and sk is secret key.
        """
        phase = encrypted[0]

        secret_key_array = self.get_sufficient_sk_power(len(encrypted))

        for j in range(1, len(encrypted)):
            for i in range(len(self._coeff_modulus)):
                phase[i] = poly_add(
                    poly_mul(encrypted[j][i], secret_key_array[j - 1][i], self._coeff_modulus[i]),
                    phase[i],
                    self._coeff_modulus[i],
                )

        return phase

    def get_sufficient_sk_power(self, max_power):
        """Generate an list of secret key polynomial raised to 1...max_power"""
        sk_power = [0] * max_power
        for i in range(1, max_power):
            sk_power[i] = [0] * len(self._coeff_modulus)  # for every polynomial in secret key.

        sk_power[0] = self._secret_key

        for i in range(2, max_power + 1):
            for j in range(len(self._coeff_modulus)):
                sk_power[i - 1][j] = poly.polypow(self._secret_key[j], i).astype(int).tolist()
        return sk_power
