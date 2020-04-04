import torch as th
from Crypto.Random import random
from torch.distributions import Normal

from syft.frameworks.torch.he.fv.util.global_variable import NOISE_STANDARD_DEVIATION


def sample_poly_ternary(parms):
    """Generate a ternary polynomial uniformlly and store in RNS representation.

    Args:
        parms: EncryptionParameters used to parametize an RNS polynomial.
    """
    coeff_modulus = parms.coeff_modulus
    coeff_count = parms.poly_modulus_degree
    coeff_mod_count = len(coeff_modulus)

    result = th.zeros(coeff_count * coeff_mod_count)
    for i in range(coeff_count):
        rand_index = random.randint(-1, 1)
        if rand_index == 1:
            for j in range(coeff_mod_count):
                result[i + j * coeff_count] = 1
        elif rand_index == -1:
            for j in range(coeff_mod_count):
                result[i + j * coeff_count] = coeff_modulus[j] - 1
        else:
            for j in range(coeff_mod_count):
                result[i + j * coeff_count] = 0
    return result


def sample_poly_normal(param):
    coeff_modulus = param.coeff_modulus
    coeff_mod_count = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree

    result = th.zeros(coeff_count * coeff_mod_count)
    for i in range(coeff_count):
        noise = Normal(th.tensor([0]), th.tensor([NOISE_STANDARD_DEVIATION]))
        noise = noise.sample()
        assert type(noise) is int
        if noise > 0:
            for j in range(coeff_mod_count):
                result[i + j * coeff_count] = noise
        elif noise < 0:
            noise = -noise
            for j in range(coeff_mod_count):
                result[i + j * coeff_count] = coeff_modulus[j] - noise
        else:
            for j in range(coeff_mod_count):
                result[i + j * coeff_count] = 0
    return result


def sample_poly_uniform(param):
    coeff_modulus = param.coeff_modulus
    coeff_mod_count = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree

    max_random = 0x7FFFFFFFFFFFFFFF
    result = th.zeros(coeff_count * coeff_mod_count)

    for j in range(coeff_mod_count):
        modulus = coeff_modulus[j]
        max_multiple = max_random - (max_random % modulus) - 1
        for i in range(coeff_count):
            # This ensures uniform distribution.
            while True:
                rand = random.getrandbits(32) << 31 | random.getrandbits(32) >> 1
                if rand < max_multiple:
                    break
            result[i + j * coeff_count] = rand % modulus
    return result


def encrypt_zero_symmetric(context, secret_key):
    param = context.param
    coeff_modulus = param.coeff_modulus
    coeff_mod_count = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree

    # (a) Sample a uniformly at random
    c1 = sample_poly_uniform(param)

    # Sample e <-- chi
    e = sample_poly_normal(param)

    # calculate -(a*s + e) (mod q) and store in c0
    c0 = -c1 * secret_key + e

    for j in range(coeff_mod_count):
        for i in range(coeff_count):
            c0[i + j * coeff_count] = c0[i + j * coeff_count] % coeff_modulus[j]

    return th.tensor([c0, c1])
