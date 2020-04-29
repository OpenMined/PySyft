from syft.frameworks.torch.he.fv.plaintext import PlainText


class Decrypter:
    """Decrypts Ciphertext objects into Plaintext objects. Constructing a Decryptor
    requires a Context object with valid encryption parameters, and the secret key.
    The Decryptor is also used to compute the invariant noise budget in a given
    ciphertext."""

    def __init__(self, context, secret_key):
        self._context = context
        self._coeff_modulus = context.param.coeff_modulus
        self._coeff_mod_size = len(self._coeff_modulus)
        self._coeff_count = context.param.poly_modulus_degree
        self._secret_key = secret_key.data
        self._plain_modulus = context.plain_modulus

    def decrypt(self, encrypted):
        pass

    def dot_product_ct_sk_array(self, encrypted):
        product = [0] * self._coeff_count * self._coeff_mod_size
        c_0, c_1 = encrypted.data

        for j in range(self._coeff_mod_size):
            for i in range(self._coeff_count):
                product[i + j * self._coeff_count] = (
                    (c_1[i + j * self._coeff_count] * self._secret_key[i + j * self._coeff_count])
                    % self._coeff_modulus[j]
                    + c_0[i + j * self._coeff_count]
                ) % self._coeff_modulus[j]

        return product  # product  = c_0 + c_1 * sk
