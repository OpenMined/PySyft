class CipherText:
    def __init__(self, data):
        self._size = 0
        self._poly_modulus_degree = 0
        self._coeff_mod_count = 0
        self._scale = 1.0
        self._data = data
