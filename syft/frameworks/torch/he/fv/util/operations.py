from syft.frameworks.torch.he.fv.ciphertext import CipherText


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


def reverse_bit(value):
    result = 0
    while value:
        result = (result << 1) + (value & 1)
        value >>= 1
    return result


def multiply_many_except(operands, count, expt):
    count
    result = 1
    for i in range(count):
        if i != expt:
            result *= operands[i]
    return result


def multiply_add_plain_with_scaling_variant(pue, message, context):
    # here pue = pk * u * e
    param = context.param
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree
    plain_coeff_count = message.coeff_count
    delta = context.coeff_div_plain_modulus
    plaintext = message.data
    pue_0, pue_1 = pue.data
    print("pue_0", type(pue_0))

    # Coefficients of plain m multiplied by coeff_modulus q, divided by plain_modulus t,
    # and rounded to the nearest integer (rounded up in case of a tie). Equivalent to
    for i in range(plain_coeff_count):
        for j in range(coeff_mod_size):
            temp = delta[j] * plaintext[i]
            pue_0[j * coeff_count] = (
                pue_0[j * coeff_count] + (temp % coeff_modulus[j])
            ) % coeff_modulus[j]
    return CipherText([pue_0, pue_1])  # p0 * u * e1 + delta * m
