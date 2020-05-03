class BaseConvertor:
    def __init__(self, ibase, obase):
        self._ibase = ibase
        self._obase = obase
        self._ibase_size = ibase.size
        self._obase_size = obase.size
        self._base_change_matrix = None

    def fast_convert_array(self, input, count):
        output = [0] * count * 2

        self._base_change_matrix = [[]] * self._obase_size

        for i in range(self._obase_size):
            self._base_change_matrix[i] = [0] * self._ibase_size

            for j in range(self._ibase_size):
                self._base_change_matrix[i][j] = (
                    self._ibase.punctured_prod_array[j] % self._obase.base[i]
                )

        temp = [0] * count * self._ibase_size
        for i in range(self._ibase_size):

            inv_ibase_punctured_prod_mod_ibase_elt = self._ibase.inv_punctured_prod_mod_base_array[
                i
            ]
            ibase_elt = self._ibase.base[i]

            for k in range(self._ibase_size):
                temp[i + k * self._ibase_size] = (
                    input[k + i * self._ibase_size] * inv_ibase_punctured_prod_mod_ibase_elt
                ) % ibase_elt

        for j in range(self._obase_size):
            obase_elt = self._obase.base[j]
            for k in range(count):
                dot_product = 0
                for tt in range(self._ibase_size):
                    dot_product += (
                        temp[tt + k * self._ibase_size] * self._base_change_matrix[j][tt]
                    ) % obase_elt
                output[k] = dot_product % obase_elt

        return output
