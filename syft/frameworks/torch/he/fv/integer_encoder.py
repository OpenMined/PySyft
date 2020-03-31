from collections import defaultdict

from syft.frameworks.torch.he.fv.modulus import PlainModulus
from syft.frameworks.torch.he.fv.plaintext import PlainText


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
        self.param = context.param
        self.plain_modulus = self.param.plain_modulus

        if self.plain_modulus <= 1:
            raise ValueError("plain_modulus must be at least 2")

        if self.plain_modulus == 2:
            # In this case we don't allow any negative numbers
            self.coeff_neg_threshold = 2
        else:
            # Normal negative threshold case
            self.coeff_neg_threshold = (self.plain_modulus + 1) >> 1

        self.neg_one = self.plain_modulus - 1

    def encode(self, value):
        """Encodes a signed integer into a plaintext polynomial.
        Args:
            value: The signed integer to encode"""

        coeff_index = 0
        if value < 0:
            # negative value.
            value = -1 * value
            encode_coeff_count = value.bit_length()
            plaintext = [0] * encode_coeff_count
            while value != 0:
                if (value & 1) != 0:
                    plaintext[coeff_index] = self.neg_one
                value >>= 1
                coeff_index += 1
        else:
            # positive value.
            encode_coeff_count = value.bit_length()
            plaintext = [0] * encode_coeff_count
            while value != 0:
                if (value & 1) != 0:
                    plaintext[coeff_index] = 1
                value >>= 1
                coeff_index += 1

        return PlainText(encode_coeff_count, plaintext)

    def decode(self, plain):
        """Decodes a plaintext polynomial and returns the result.
        Mathematically this amounts to evaluating the input polynomial at x=2.

        Args:
            plain: The plaintext to be decoded"""

        result = 0
        bit_index = plain.significant_coeff_count()
        while bit_index > 0:
            bit_index -= 1
            coeff = plain.data[bit_index]

            # Left shift result.
            next_result = result << 1
            if (next_result < 0) != (result < 0):
                # Check for overflow.
                raise OverflowError("output out of range")

            if coeff >= self.plain_modulus:
                # Coefficient is bigger than plaintext modulus
                raise ValueError("plain does not represent a valid plaintext polynomial")

            coeff_is_negative = coeff >= self.coeff_neg_threshold
            pos_value = coeff

            if coeff_is_negative:
                pos_value = self.plain_modulus - pos_value

            coeff_value = pos_value
            if coeff_is_negative:
                coeff_value = -coeff_value

            next_result_was_negative = next_result < 0
            next_result += coeff_value
            next_result_is_negative = next_result < 0
            if (
                next_result_was_negative == coeff_is_negative
                and next_result_was_negative != next_result_is_negative
            ):
                # Accumulation and coefficient had same signs, but accumulator changed signs after addition, so must be overflow.
                raise OverflowError("output out of range")
            result = next_result
        return result
