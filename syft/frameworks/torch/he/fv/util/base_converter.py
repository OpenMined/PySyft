from syft.frameworks.torch.he.fv.util.operations import multiply_mod


class BaseConvertor:
    def __init__(self, ibase, obase):
        self._ibase = ibase
        self._obase = obase

        self._base_change_matrix = [[]] * self._obase.size
        for i in range(self._obase.size):
            self._base_change_matrix[i] = [0] * self._ibase.size

            for j in range(self._ibase.size):
                self._base_change_matrix[i][j] = (
                    self._ibase.punctured_prod_array[j] % self._obase.base[i]
                )

    def fast_convert_array(self, input, count):
        output = [0] * count * self._obase.size

        temp = [0] * count * self._ibase.size
        for i in range(self._ibase.size):
            inv_ibase_punctured_prod_mod_ibase = self._ibase.inv_punctured_prod_mod_base_array[i]
            ibase = self._ibase.base[i]

            for k in range(count):
                temp[i + k * self._ibase.size] = multiply_mod(
                    input[k + i * count], inv_ibase_punctured_prod_mod_ibase, ibase
                )

        for j in range(self._obase.size):
            obase = self._obase.base[j]

            for k in range(count):
                dot_product = 0

                for tt in range(self._ibase.size):
                    dot_product += multiply_mod(
                        temp[tt + k * self._ibase.size], self._base_change_matrix[j][tt], obase
                    )
                output[k + j * count] = dot_product % obase

        return output
