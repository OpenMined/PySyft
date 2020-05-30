import numpy as np
from numpy.polynomial import polynomial as poly

from syft.frameworks.torch.he.fv.ciphertext import CipherText


def multiply_mod(operand1, operand2, modulus):
    return (operand1 * operand2) % modulus


def negate_mod(operand, modulus):
    """returns (-1 * operand) % modulus"""
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


def poly_add_mod(op1, op2, modulus):
    """return addition of two polynomials with all coefficients of
    polynomial %q(coefficient modulus)"""
    return np.mod(np.polyadd(op1, op2), modulus).tolist()


def poly_mul_mod(op1, op2, modulus):
    """return multiplication of two polynomials with all coefficients of
    polynomial %q(coefficient modulus) and result polynomial % t(polynomial modulus)"""
    poly_mod = np.array([1] + [0] * (len(op1) - 1) + [1])
    result = (
        poly.polydiv(
            poly.polymul(np.array(op1, dtype="object"), np.array(op2, dtype="object")) % modulus,
            poly_mod,
        )[1]
        % modulus
    ).tolist()
    return [round(x) for x in result]


def poly_negate_mod(op, modulus):
    """returns negative of polynomial i.e (-1 * op)"""
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
    """removes leading zero's from the list."""
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
    """ Extended GCD

    Args:
        x (integer)
        y (integer)

    Returns:
        (gcd, x, y) where gcd is the greatest common divisor of a and b.
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
    """Add message (PlainText) into phase.

    Args:
        phase: phase is pre-computed carrier polynomial where we can add message data.
        message (Plaintext): A plaintext representation of integer data to be encrypted.
        context (Context): Context for extracting encryption parameters.

    Returns:
        A Ciphertext object with the encrypted result of encryption process.
    """
    coeff_modulus = context.param.coeff_modulus
    message = message.data
    plain_coeff_count = len(message)
    delta = context.coeff_div_plain_modulus
    phase0, phase1 = phase.data  # here phase = pk * u * e

    # Coefficients of plain m multiplied by coeff_modulus q, divided by plain_modulus t,
    # and rounded to the nearest integer (rounded up in case of a tie). Equivalent to
    for i in range(plain_coeff_count):
        for j in range(len(coeff_modulus)):
            temp = round(delta[j] * message[i]) % coeff_modulus[j]
            phase0[j][i] = (phase0[j][i] + temp) % coeff_modulus[j]

    return CipherText([phase0, phase1])  # phase0 = pk0 * u * e + delta * m
