from syft.frameworks.torch.he.fv.ciphertext import CipherText


def multiply_mod(operand1, operand2, modulus):
    return (operand1 * operand2) % modulus


def add_mod(operand1, operand2, modulus):
    return (operand1 + operand2) % modulus


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


def get_significant_count(values, count):
    i = count - 1
    while count and not values[i]:
        i -= 1
        count -= 1
    return count


def reverse_bit(value):
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


def negate_int_mod(operand, modulus):
    if modulus == 0:
        raise ValueError("Modulus cannot be 0")
    if operand >= modulus:
        raise OverflowError("operand cannot be greater than modulus")
    non_zero = operand != 0
    return (modulus - operand) & (-int(non_zero))


def xgcd(x, y):
    """ Extended GCD:
        Returns (gcd, x, y) where gcd is the greatest common divisor of a and b.
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


def try_invert_int_mod(value, modulus):
    if value == 0:
        return False
    gcd_tuple = xgcd(value, modulus)

    if gcd_tuple[1] < 0:
        return gcd_tuple[1] + modulus
    else:
        return gcd_tuple[1]


def multiply_add_plain_with_scaling_variant(pue, message, context):
    # here pue = pk * u * e
    coeff_modulus = context.param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = context.param.poly_modulus_degree
    plain_coeff_count = message.coeff_count
    delta = context.coeff_div_plain_modulus
    message = message.data
    pue_0, pue_1 = pue.data

    # Coefficients of plain m multiplied by coeff_modulus q, divided by plain_modulus t,
    # and rounded to the nearest integer (rounded up in case of a tie). Equivalent to
    for i in range(plain_coeff_count):
        for j in range(coeff_mod_size):
            temp = round(delta[j] * message[i]) % coeff_modulus[j]
            pue_0[i + j * coeff_count] = (pue_0[i + j * coeff_count] + temp) % coeff_modulus[j]

    return CipherText([pue_0, pue_1])  # p0 * u * e1 + delta * m
