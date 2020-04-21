import numpy as np
from syft.frameworks.torch.he.fv.util.base_convertor import BaseConvertor


class Context:
    """Stores a set of attributes (qualifiers) of a set of encryption parameters.
    These parameters are mainly used internally in various parts of the library,
    e.g., to determine which algorithmic optimizations the current support. The
    qualifiers are automatically created by the FVContext class, silently passed
    on to classes such as Encryptor, Evaluator, and Decryptor, and the only way to
    change them is by changing the encryption parameters themselves. In other
    words, a user will never have to create their own instance of this class, and
    in most cases never have to worry about it at all.
    """

    def __init__(self, encryption_param):
        self._param = encryption_param
        self._poly_modulus_degree = self._param.poly_modulus_degree
        self._coeff_modulus = self._param.coeff_modulus
        self._plain_modulus = encryption_param.plain_modulus
        self._coeff_div_plain_modulus = [int(x / self._plain_modulus) for x in self._coeff_modulus]
        self._plain_div_coeff_modulus = [self._plain_modulus / x for x in self._coeff_modulus]
        # self._base_converter = BaseConvertor(self._coeff_modulus, self._poly_modulus_degree, self._plain_modulus)

    @property
    def param(self):
        return self._param

    @property
    def plain_modulus(self):
        return self._plain_modulus

    @property
    def coeff_div_plain_modulus(self):
        return self._coeff_div_plain_modulus

    @property
    def plain_div_coeff_modulus(self):
        return self._plain_div_coeff_modulus

    @property
    def base_convertor(self):
        return self._base_converter
