import random


def sample_poly_ternary(parms):
    """Generate a ternary polynomial uniformlly and store in RNS representation.

    Args:
        parms: EncryptionParameters used to parametize an RNS polynomial.
    """
    coeff_modulus = parms.coeff_modulus
    coeff_count = parms.poly_modulus_degree
    coeff_mod_count = len(coeff_modulus)

    destination = [0] * coeff_count * coeff_mod_count

    for i in range(coeff_count):
        rand_index = random.randint(-1, 1)
        if rand_index == 1:
            for j in range(coeff_mod_count):
                destination[i + j * coeff_count] = 1
        elif rand_index == -1:
            for j in range(coeff_mod_count):
                destination[i + j * coeff_count] = coeff_modulus[j] - 1
        else:
            for j in range(coeff_mod_count):
                destination[i + j * coeff_count] = 0

    return destination
