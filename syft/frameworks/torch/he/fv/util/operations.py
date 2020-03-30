def multiply_mod(operand1, operand2, modulus):
    return (operand1 * operand2) % modulus


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


def get_significant_bit_count(value):

    if value == 0:
        return 0

    deBruijnTable64 = [
        63,
        0,
        58,
        1,
        59,
        47,
        53,
        2,
        60,
        39,
        48,
        27,
        54,
        33,
        42,
        3,
        61,
        51,
        37,
        40,
        49,
        18,
        28,
        20,
        55,
        30,
        34,
        11,
        43,
        14,
        22,
        4,
        62,
        57,
        46,
        52,
        38,
        26,
        32,
        41,
        50,
        36,
        17,
        19,
        29,
        10,
        13,
        21,
        56,
        45,
        25,
        31,
        35,
        16,
        9,
        12,
        44,
        24,
        15,
        8,
        23,
        7,
        6,
        5,
    ]
    value |= value >> 1
    value |= value >> 2
    value |= value >> 4
    value |= value >> 8
    value |= value >> 16
    value |= value >> 32

    return deBruijnTable64[((value - (value >> 1)) * 0x07EDD5E59A4E28C2) >> 58] + 1
