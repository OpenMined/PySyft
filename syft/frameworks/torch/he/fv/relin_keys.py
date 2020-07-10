from syft.frameworks.torch.he.fv.util.operations import poly_mul_mod
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_symmetric
from syft.frameworks.torch.he.fv.util.operations import poly_add_mod


class RelinKeys:
    def __init__(self, context, sk):
        self._context = context
        self._coeff_modulus = context.param.coeff_modulus
        self._poly_modulus = context.param.poly_modulus
        self._secret_key = sk.data
        self._secret_key_power_2 = self._get_sk_power_2(sk.data)

    def _generate_relin_key(self):
        return self._generate_one_kswitch_key(self._secret_key_power_2)

    def _generate_one_kswitch_key(self, sk_power_2):
        result = [0] * len(self._coeff_modulus)
        for i in range(len(self._coeff_modulus)):
            result[i] = encrypt_symmetric(self._context, self._secret_key).data
            factor = self._coeff_modulus[-1] % self._coeff_modulus[i]

            temp = [(x * factor) for x in sk_power_2[i]]

            for j in range(len(self._coeff_modulus)):
                result[i][0][j] = poly_add_mod(
                    result[i][0][j], temp, self._coeff_modulus[j], self._poly_modulus
                )
        return result

    def _get_sk_power_2(self, sk):
        sk_power_2 = None
        for i in range(len(self._coeff_modulus)):
            sk_power_2 = poly_mul_mod(sk[i], sk[i], self._coeff_modulus[i], self._poly_modulus,)
        return sk_power_2
