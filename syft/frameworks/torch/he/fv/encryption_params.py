from logging import warning


class EncryptionParams:
    """Represents user-customizable encryption scheme settings. The parameters (most
    importantly poly_modulus, coeff_modulus, plain_modulus) significantly affect
    the performance, capabilities, and security of the encryption scheme. Once
    an instance of EncryptionParameters is populated with appropriate parameters,
    it can be used to create an instance of the FVContext class, which verifies
    the validity of the parameters, and performs necessary pre-computations.
    """

    def __init__(self, poly_modulus_degree=2, coeff_modulus=[], plain_modulus=0):
        self._poly_modulus_degree = poly_modulus_degree
        self._coeff_modulus = coeff_modulus
        self._plain_modulus = plain_modulus

    @property
    def poly_modulus_degree(self):
        return self._poly_modulus_degree

    @poly_modulus_degree.setter
    def poly_modulus_degree(self, value):
        """
            Sets the degree of the polynomial modulus parameter to the specified value.
            The polynomial modulus directly affects the number of coefficients in
            plaintext polynomials, the size of ciphertext elements, the computational
            performance of the scheme (bigger is worse), and the security level (bigger
            is better). In general the degree of the polynomial modulus should be
            a power of 2 (e.g.  1024, 2048, 4096, 8192, 16384, or 32768).
        """
        if value >= 2:
            if (value & (value - 1) == 0) and value != 0:
                self._poly_modulus_degree = value
            else:
                raise ValueError("poly_modulus_degree must be a power of two 2")
        else:
            raise ValueError("poly_modulus_degree must be at least 2")

    @property
    def coeff_modulus(self):
        return self._coeff_modulus

    @coeff_modulus.setter
    def coeff_modulus(self, value):
        """
            Sets the coefficient modulus parameter. The coefficient modulus consists
            of a list of distinct prime numbers, and is represented as a list.
            The coefficient modulus directly affects the size of ciphertext elements,
            the amount of computation that the scheme can perform (bigger is better),
            and the security level (bigger is worse).
        """
        self._coeff_modulus = value

    @property
    def plain_modulus(self):
        return self._plain_modulus

    @plain_modulus.setter
    def plain_modulus(self, value):
        """
            Sets the plaintext modulus parameter. The plaintext modulus
            determines the largest coefficient that plaintext polynomials can represent.
            It also affects the amount of computation that the scheme can perform
            (bigger is worse).
        """
        self._plain_modulus = value
