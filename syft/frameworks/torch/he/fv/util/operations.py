import numpy as np
from numpy.polynomial import polynomial as poly

from syft.frameworks.torch.he.fv.ciphertext import CipherText


def multiply_mod(operand1, operand2, modulus):
    return (operand1 * operand2) % modulus


def add_mod(operand1, operand2, modulus):
    return (operand1 + operand2) % modulus


def negate_mod(operand, modulus):
    # returns (-1 * operand) % modulus
    if modulus == 0:
        raise ValueError("Modulus cannot be 0")
    if operand >= modulus:
        raise OverflowError("operand cannot be greater than modulus")
    non_zero = operand != 0
    return (modulus - operand) & (-int(non_zero))


def exponentiate_mod(operand, exponent, modulus):
    if exponent == 0:
        return 1

    if exponent == 1:
        return operand

    # Perform binary exponentiation.
    power = operand
    product = 0
    intermediate = 1

    # Initially: power = operand and intermediate = 1, product is irrelevant.
    while True:
        if exponent & 1:
            product = multiply_mod(power, intermediate, modulus)
            product, intermediate = intermediate, product

        exponent >>= 1

        if exponent == 0:
            break

        product = multiply_mod(power, power, modulus)
        product, power = power, product

    return intermediate


def invert_mod(value, modulus):
    """calculate inverse modulus for given value and modulus"""
    gcd_tuple = xgcd(value, modulus)

    if gcd_tuple[1] < 0:
        return gcd_tuple[1] + modulus
    else:
        return gcd_tuple[1]


def poly_add(op1, op2, modulus):
    return np.mod(np.polyadd(op1, op2), modulus).tolist()


def poly_mul(op1, op2, modulus):
    poly_mod = len(op1)
    return np.int64(np.round(poly.polydiv(poly.polymul(op1, op2) % modulus, poly_mod)[1] % modulus))


def poly_negate(op, modulus):

    coeff_count = len(op)

    result = [0] * coeff_count
    for i in range(coeff_count):
        if modulus == 0:
            raise ValueError("Modulus cannot be 0")
        if op[i] >= modulus:
            raise OverflowError("operand cannot be greater than modulus")
        non_zero = op[i] != 0
        result[i] = (modulus - op[i]) & (-int(non_zero))
    return result


def get_significant_count(values):
    """removes all leading zero from the list."""
    count = len(values)
    i = count - 1
    while count and not values[i]:
        i -= 1
        count -= 1
    return count


def reverse_bit(value):
    """calculate the value of the reverse binary representation of the given integer."""
    result = 0
    while value:
        result = (result << 1) + (value & 1)
        value >>= 1
    return result


def multiply_many_except(operands, count, expt):
    result = 1
    for i in range(count):
        if i != expt:
            result *= operands[i]
    return result


def xgcd(x, y):
    """ Extended GCD:
        returns (gcd, x, y) where gcd is the greatest common divisor of a and b.
        The numbers x, y are such that gcd = ax + by.
    """
    prev_a = 1
    a = 0
    prev_b = 0
    b = 1

    while y != 0:
        q = x // y
        temp = x % y
        x = y
        y = temp

        temp = a
        a = prev_a - q * a
        prev_a = temp

        temp = b
        b = prev_b - q * b
        prev_b = temp
    return [x, prev_a, prev_b]


def multiply_add_plain_with_delta(phase, message, context):
    """Add message(plaintext) into phase.

    Args:
        phase: phase is the result of (public_key * u + e)
        message: A Plaintext object of integer to be encrypted.
        context: A Context object for supplying encryption parameters.

    Returns:
        A Ciphertext object with the encrypted result of encryption process.
    """
    coeff_modulus = context.param.coeff_modulus
    plain_coeff_count = message.coeff_count
    delta = context.coeff_div_plain_modulus
    message = message.data
    phase0, phase1 = phase.data  # here phase = pk * u * e

    # Coefficients of plain m multiplied by coeff_modulus q, divided by plain_modulus t,
    # and rounded to the nearest integer (rounded up in case of a tie). Equivalent to
    for i in range(plain_coeff_count):
        for j in range(len(coeff_modulus)):
            temp = round(delta[j] * message[i]) % coeff_modulus[j]
            phase0[j][i] = (phase0[j][i] + temp) % coeff_modulus[j]

    return CipherText([phase0, phase1])  # phase0 = pk0 * u * e + delta * m
