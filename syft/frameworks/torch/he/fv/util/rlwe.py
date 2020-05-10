import torch as th
from secrets import SystemRandom
from secrets import randbits
from torch.distributions import Normal

from syft.frameworks.torch.he.fv.ciphertext import CipherText
from syft.frameworks.torch.he.fv.util.global_variable import NOISE_STANDARD_DEVIATION


def sample_poly_ternary(parms):
    """Generate a ternary polynomial uniformlly and store in RNS representation.

    Args:
        parms: EncryptionParameters used to parametize an RNS polynomial.
    """
    coeff_modulus = parms.coeff_modulus
    coeff_count = parms.poly_modulus_degree
    coeff_mod_size = len(coeff_modulus)

    result = [0] * coeff_count * coeff_mod_size
    for i in range(coeff_count):
        r = SystemRandom()
        rand_index = r.choice([-1, 0, 1])
        if rand_index == 1:
            for j in range(coeff_mod_size):
                result[i + j * coeff_count] = 1
        elif rand_index == -1:
            for j in range(coeff_mod_size):
                result[i + j * coeff_count] = coeff_modulus[j] - 1
        else:
            for j in range(coeff_mod_size):
                result[i + j * coeff_count] = 0
    return result


def sample_poly_normal(param):
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree

    result = [0] * coeff_count * coeff_mod_size
    for i in range(coeff_count):
        noise = Normal(th.tensor([0.0]), th.tensor(NOISE_STANDARD_DEVIATION))
        noise = int(noise.sample().item())
        if noise > 0:
            for j in range(coeff_mod_size):
                result[i + j * coeff_count] = noise
        elif noise < 0:
            noise = -noise
            for j in range(coeff_mod_size):
                result[i + j * coeff_count] = coeff_modulus[j] - noise
        else:
            for j in range(coeff_mod_size):
                result[i + j * coeff_count] = 0
    return result


def sample_poly_uniform(param):
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree

    max_random = 0x7FFFFFFFFFFFFFFF
    result = [0] * coeff_count * coeff_mod_size

    for j in range(coeff_mod_size):
        modulus = coeff_modulus[j]
        max_multiple = max_random - (max_random % modulus) - 1
        for i in range(coeff_count):
            # This ensures uniform distribution.
            while True:
                rand = randbits(32) << 31 | randbits(32) >> 1
                if rand < max_multiple:
                    break
            result[i + j * coeff_count] = rand % modulus
    return result


def encrypt_zero_asymmetric(context, public_key):
    param = context.param
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree
    public_key = public_key.data
    encrypted_size = 2

    # Generate u <-- R_3
    u = sample_poly_ternary(param)

    # c[j] = u * public_key[j]
    c_0 = [0] * coeff_count * coeff_mod_size
    c_1 = [0] * coeff_count * coeff_mod_size
    result = [c_0, c_1]

    for k in range(encrypted_size):
        for j in range(coeff_mod_size):
            for i in range(coeff_count):
                result[k][i + j * coeff_count] = (
                    u[i + j * coeff_count] * public_key[k][i + j * coeff_count]
                ) % coeff_modulus[j]

    # Generate e_j <-- chi
    # c[j] = public_key[j] * u + e[j]
    for k in range(encrypted_size):
        e = sample_poly_normal(param)
        for j in range(coeff_mod_size):
            for i in range(coeff_count):
                result[k][i + j * coeff_count] = (
                    result[k][i + j * coeff_count] + e[i + j * coeff_count]
                ) % coeff_modulus[j]

    return CipherText(result)  # result = public_key[j] * u + e[j]


def encrypt_zero_symmetric(context, secret_key):
    param = context.param
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree
    secret_key = secret_key.data

    # (a) Sample a uniformly at random
    c1 = sample_poly_uniform(param)

    # Sample e <-- chi
    e = sample_poly_normal(param)

    # calculate -(a*s + e) (mod q) and store in c0
    c0 = [0] * coeff_count * coeff_mod_size
    for j in range(coeff_mod_size):
        for i in range(coeff_count):
            c0[i + j * coeff_count] = (
                (c1[i + j * coeff_count] * secret_key[i + j * coeff_count]) % coeff_modulus[j]
                + e[i + j * coeff_count]
            ) % coeff_modulus[j]

    for j in range(coeff_mod_size):
        for i in range(coeff_count):
            non_zero = c0[i + j * coeff_count] != 0
            c0[i + j * coeff_count] = (coeff_modulus[j] - c0[i + j * coeff_count]) & (
                -int(non_zero)
            )

    return CipherText([c0, c1])
