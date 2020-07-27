from syft.frameworks.torch.he.fv.util.operations import multiply_mod


class BaseConvertor:
    """BaseConvertor is used for converting plain/base of integer
    from one set of bases to another set of bases.

    Args:
        ibase: A list of integer denoting the input base.
        obase: A list of integer denoting the base of the output required.
    """

    def __init__(self, ibase, obase):
        self._ibase = ibase
        self._obase = obase

        # base_change_matrix is helpful for fast conversion as many pre-computation
        # are already done here only once.
        self._base_change_matrix = [[]] * self._obase.size
        for i in range(self._obase.size):
            self._base_change_matrix[i] = [0] * self._ibase.size

            for j in range(self._ibase.size):
                self._base_change_matrix[i][j] = (
                    self._ibase.punctured_prod_list[j] % self._obase.base[i]
                )

    def fast_convert_list(self, input, count):
        """Converts the plain/base of input list from input base to output base
        declared at the time of initialization of BaseConvertor class.

        Args:
            input: A list of integers needed to be converted from input base to output base.
            count: An integer denoting the coefficient count of output base.

        Returns:
            A list of integers converted from input base plain to output base plain.
        """
        output = [[0] * count for i in range(self._obase.size)]

        temp = [[0] * count for i in range(self._ibase.size)]

        for i in range(self._ibase.size):
            inv_punctured_prod_mod_ibase = self._ibase.inv_punctured_prod_mod_base_list[i]
            ibase = self._ibase.base[i]

            for k in range(count):
                temp[i][k] = multiply_mod(input[i][k], inv_punctured_prod_mod_ibase, ibase)

        for j in range(self._obase.size):
            obase = self._obase.base[j]

            for k in range(count):
                dot_product = 0

                for tt in range(self._ibase.size):
                    dot_product += multiply_mod(temp[tt][k], self._base_change_matrix[j][tt], obase)
                output[j][k] = dot_product % obase

        return output
