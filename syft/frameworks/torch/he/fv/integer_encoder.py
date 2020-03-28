from syft.frameworks.torch.he.fv.util.operations import get_significant_bit_count


class IntegerEncoder:
    """Encodes integers into plaintext polynomials that Encryptor can encrypt. An instance of
    the IntegerEncoder class converts an integer into a plaintext polynomial by placing its
    binary digits as the coefficients of the polynomial. Decoding the integer amounts to
    evaluating the plaintext polynomial at x=2.

    Addition and multiplication on the integer side translate into addition and multiplication
    on the encoded plaintext polynomial side, provided that the length of the polynomial
    never grows to be of the size of the polynomial modulus (poly_modulus), and that the
    coefficients of the plaintext polynomials appearing throughout the computations never
    experience coefficients larger than the plaintext modulus (plain_modulus).

    Negative integers are represented by using -1 instead of 1 in the binary representation,
    and the negative coefficients are stored in the plaintext polynomials as unsigned integers
    that represent them modulo the plaintext modulus. Thus, for example, a coefficient of -1
    would be stored as a polynomial coefficient plain_modulus-1."""

    def __init__(self, context):
        self.param = context.param()
        self.plain_modulus = self.param.plain_modulus()

        if self.plain_modulus <= 1:
            raise ValueError("plain_modulus must be at least 2")

        if self.plain_modulus == 2:
            # In this case we don't allow any negative numbers
            self.coeff_neg_threshold_ = 2
        else:
            # Normal negative threshold case
            self.coeff_neg_threshold_ = (self.plain_modulus + 1) >> 1

        self.neg_one = self.plain_modulus - 1

    def encode(self, value):
        """Encodes a signed integer into a plaintext polynomial.
        Args:
            value: The signed integer to encode"""

        plaintext = []
        coeff_index = 0
        if value < 0:
            # negative value.
            pos_value = -1 * value
            encode_coeff_count = get_significant_bit_count(pos_value)
            while pos_value != 0:
                if (pos_value & 1) != 0:
                    plaintext[coeff_index] = self.neg_one
                pos_value >>= 1
                coeff_index += 1
        else:
            # positive value.
            encode_coeff_count = get_significant_bit_count(value)
            while value != 0:
                if (value & 1) != 0:
                    plaintext[coeff_index] = 1
                value >>= 1
                coeff_index += 1

        return plaintext

    def decode(self, plain):
        """Decodes a plaintext polynomial and returns the result.
        Mathematically this amounts to evaluating the input polynomial at x=2.

        Args:
            plain: The plaintext to be decoded"""
        pass
