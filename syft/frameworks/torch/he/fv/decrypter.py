from syft.frameworks.torch.he.fv.plaintext import PlainText


class Decrypter:
    """Decrypts Ciphertext objects into Plaintext objects. Constructing a Decryptor
    requires a Context object with valid encryption parameters, and the secret key.
    The Decryptor is also used to compute the invariant noise budget in a given
    ciphertext."""

    def __init__(self, context, secret_key):
        self._context = context
        self._coeff_modulus = context.param.coeff_modulus
        self._coeff_mod_count = len(self._coeff_modulus)
        self._coeff_count = context.param.poly_modulus_degree
        self._secret_key = secret_key.data
        self._plain_modulus = context.plain_modulus

    def decrypt(self, ciphertext):
        t_div_q = self._context.plain_div_coeff_modulus
        result = [0] * self._coeff_count * self._coeff_mod_count
        c_0, c_1 = ciphertext.data

        # print('coeff_mod_count : ',self._coeff_mod_count)
        # print('coeff_count : ', self._coeff_count)
        # print("size of c1 : ",c_1.size())
        # print('size of sk : ', self._secret_key.size())

        for j in range(self._coeff_mod_count):
            for i in range(self._coeff_count):
                result[i + j * self._coeff_count] = (
                    (
                        (
                            (
                                c_1[i + j * self._coeff_count]
                                * self._secret_key[i + j * self._coeff_count]
                            )
                            % self._coeff_modulus[j]
                            + c_0[i + j * self._coeff_count]
                        )
                        % self._coeff_modulus[j]
                    )
                    * t_div_q[j]
                ).round() % self._plain_modulus

        for j in range(self._coeff_mod_count):
            for i in range(self._coeff_count):
                result[i + j * self._coeff_count] = round(result[i + j * self._coeff_count].item())
        return PlainText(result)
