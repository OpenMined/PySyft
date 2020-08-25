class EncryptionParams:
    """A class to hold the encryption parameters for easy access by any component of scheme.

    Attribute:
        poly_modulus: The polynomial modulus directly affects the number of coefficients in
            plaintext polynomials, the size of ciphertext elements, the computational
            performance of the scheme (bigger is worse), and the security level (bigger
            is better). In general the degree of the polynomial modulus should be
            a power of 2 (e.g.  1024, 2048, 4096, 8192, 16384, or 32768).

        coeff_modulus: The coefficient modulus consists of a list of distinct prime numbers,
            and is represented as a list. The coefficient modulus directly affects the size
            of ciphertext elements, the amount of computation that the scheme can perform
            (bigger is better), and the security level (bigger is worse).

        plain_modulus: The plaintext modulus determines the largest coefficient that plaintext
            polynomials can represent. It also affects the amount of computation that the scheme
            can perform (bigger is worse).
    """

    def __init__(self, poly_modulus, coeff_modulus, plain_modulus):

        if poly_modulus >= 2:
            if (poly_modulus & (poly_modulus - 1) == 0) and poly_modulus != 0:
                self.poly_modulus = poly_modulus
            else:
                raise ValueError("poly_modulus must be a power of two 2")
        else:
            raise ValueError("poly_modulus must be at least 2")

        self.coeff_modulus = coeff_modulus

        self.plain_modulus = plain_modulus
        self.param_id = self.compute_parms_id()

    def set_coeff_modulus(self, coeff_mod):
        """Set coefficient modulus and generate new param_id."""
        self.coeff_modulus = coeff_mod
        self.param_id = self.compute_parms_id()

    def compute_parms_id(self):
        # TODO: use hash function here.
        param_data = [self.poly_modulus, *self.coeff_modulus, self.plain_modulus]
        return " ".join(str(x) for x in param_data)
