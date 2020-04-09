import numpy as np


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
        self._total_coeff_modulus = np.prod(self._param.coeff_modulus)
        self._plain_moduls = encryption_param.plain_modulus
        self._coeff_div_plain_modulus = int(self._total_coeff_modulus / self._plain_moduls)
        self._coeff_mod_plain_modulus = self._total_coeff_modulus % self._plain_moduls

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, value):
        self._param = value

    @property
    def coeff_div_plain_modulus(self):
        return self._coeff_div_plain_modulus

    @coeff_div_plain_modulus.setter
    def coeff_div_plain_modulus(self, value):
        self._coeff_div_plain_modulus = value

    @property
    def coeff_mod_plain_modulus(self):
        return self._coeff_mod_plain_modulus

    @coeff_div_plain_modulus.setter
    def coeff_div_plain_modulus(self, value):
        self._coeff_mod_plain_modulus = value
