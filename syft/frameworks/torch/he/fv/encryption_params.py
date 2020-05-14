class EncryptionParams:
    """A class for holding all the encryption parameters at one place and for easy supplying
    of the encryption params to any component of scheme.
    """

    def __init__(self, poly_modulus_degree, coeff_modulus, plain_modulus):
        """
            Sets the degree of the polynomial modulus parameter to the specified value.
            The polynomial modulus directly affects the number of coefficients in
            plaintext polynomials, the size of ciphertext elements, the computational
            performance of the scheme (bigger is worse), and the security level (bigger
            is better). In general the degree of the polynomial modulus should be
            a power of 2 (e.g.  1024, 2048, 4096, 8192, 16384, or 32768).
        """
        if poly_modulus_degree >= 2:
            if (poly_modulus_degree & (poly_modulus_degree - 1) == 0) and poly_modulus_degree != 0:
                self.poly_modulus_degree = poly_modulus_degree
            else:
                raise ValueError("poly_modulus_degree must be a power of two 2")
        else:
            raise ValueError("poly_modulus_degree must be at least 2")

        """
            Sets the coefficient modulus parameter. The coefficient modulus consists
            of a list of distinct prime numbers, and is represented as a list.
            The coefficient modulus directly affects the size of ciphertext elements,
            the amount of computation that the scheme can perform (bigger is better),
            and the security level (bigger is worse).
        """
        self.coeff_modulus = coeff_modulus

        """
            Sets the plaintext modulus parameter. The plaintext modulus
            determines the largest coefficient that plaintext polynomials can represent.
            It also affects the amount of computation that the scheme can perform
            (bigger is worse).
        """
        self.plain_modulus = plain_modulus
