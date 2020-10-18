from syft.frameworks.torch.he.fv.util.operations import poly_mul_mod
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_symmetric
from syft.frameworks.torch.he.fv.util.operations import poly_add_mod


class RelinKey:
    def __init__(self, data):
        self.data = data


class RelinKeys:
    def __init__(self, context, sk):
        self._context = context
        key_param = context.context_data_map[context.key_param_id].param
        self._coeff_modulus = key_param.coeff_modulus
        self._poly_modulus = key_param.poly_modulus
        self._secret_key = sk.data
        self._secret_key_power_2 = self._get_sk_power_2(sk.data)

    def _generate_relin_key(self):
        return RelinKey(self._generate_one_kswitch_key(self._secret_key_power_2))

    def _generate_one_kswitch_key(self, sk_power_2):
        decomp_mod_count = len(self._coeff_modulus) - 1
        result = [0] * (decomp_mod_count)
        for i in range(decomp_mod_count):
            result[i] = encrypt_symmetric(
                self._context, self._context.key_param_id, self._secret_key
            ).data
            factor = self._coeff_modulus[-1] % self._coeff_modulus[i]

            temp = [(x * factor) for x in sk_power_2[i]]

            result[i][0][i] = poly_add_mod(
                result[i][0][i], temp, self._coeff_modulus[i], self._poly_modulus
            )
        return result

    def _get_sk_power_2(self, sk):
        sk_power_2 = []
        for i in range(len(self._coeff_modulus)):
            sk_power_2.append(
                poly_mul_mod(sk[i], sk[i], self._coeff_modulus[i], self._poly_modulus)
            )
        return sk_power_2
