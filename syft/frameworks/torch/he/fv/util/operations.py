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
