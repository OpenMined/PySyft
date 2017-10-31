class PrecisionConfig(object):

    def __init__(self):

        self.BASE = 10

        self.PRECISION_INTEGRAL = 8
        self.PRECISION_FRACTIONAL = 8
        self.Q = 293973345475167247070445277780365744413

        self.PRECISION = self.PRECISION_INTEGRAL + self.PRECISION_FRACTIONAL

        assert(self.Q > self.BASE**self.PRECISION)

        self.INVERSE = 104491423396290281423421247963055991507  # inverse of BASE**FRACTIONAL_PRECISION
        self.KAPPA = 6  # leave room for five digits overflow before leakage

        assert((self.INVERSE * self.BASE**self.PRECISION_FRACTIONAL) % self.Q == 1)
        assert(self.Q > self.BASE**(2 * self.PRECISION + self.KAPPA))
