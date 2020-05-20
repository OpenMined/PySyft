import torch as th
from secrets import SystemRandom
from secrets import randbits
from torch.distributions import Normal

from syft.frameworks.torch.he.fv.ciphertext import CipherText
from syft.frameworks.torch.he.fv.util.operations import add_mod
from syft.frameworks.torch.he.fv.util.operations import multiply_mod
from syft.frameworks.torch.he.fv.util.operations import negate_mod
from syft.frameworks.torch.he.fv.util.global_variable import NOISE_STANDARD_DEVIATION


def sample_poly_ternary(parms):
    """Generate a ternary polynomial uniformally with elements [-1, 0, 1] where -1 is represented as (modulus - 1)
    because -1 % modulus == modulus - 1.

    Used for generating secret key using coeff_modulus(list of prime nos) which represents as 'q' in the research paper.

    Args:
        parms: A valid EncryptionParam class object required for extracting the encryption parameters.
    """
    coeff_modulus = parms.coeff_modulus
    coeff_count = parms.poly_modulus_degree
    coeff_mod_size = len(coeff_modulus)

    result = [0] * coeff_mod_size
    for i in range(coeff_mod_size):
        result[i] = [0] * coeff_count

    for i in range(coeff_count):
        rand_index = SystemRandom().choice([-1, 0, 1])
        if rand_index == 1:
            for j in range(coeff_mod_size):
                result[j][i] = 1
        elif rand_index == -1:
            for j in range(coeff_mod_size):
                result[j][i] = coeff_modulus[j] - 1
        else:
            for j in range(coeff_mod_size):
                result[j][i] = 0
    return result


def sample_poly_normal(param):
    """Generate a polynomial from normal distribution where negative values are represented as
    (modulus - value) a positive value.

    Args:
        parms: A valid EncryptionParam class object required for extracting the encryption parameters.
    """
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree

    result = [0] * coeff_mod_size
    for i in range(coeff_mod_size):
        result[i] = [0] * coeff_count

    for i in range(coeff_count):
        noise = Normal(th.tensor([0.0]), th.tensor(NOISE_STANDARD_DEVIATION))
        noise = int(noise.sample().item())
        if noise > 0:
            for j in range(coeff_mod_size):
                result[j][i] = noise
        elif noise < 0:
            noise = -noise
            for j in range(coeff_mod_size):
                result[j][i] = coeff_modulus[j] - noise
        else:
            for j in range(coeff_mod_size):
                result[j][i] = 0
    return result


def sample_poly_uniform(param):
    """Generate a polynomial from uniform distribution.

    Args:
        parms: A valid EncryptionParam class object required for extracting the encryption parameters.
    """
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree

    max_random = 0x7FFFFFFFFFFFFFFF
    result = [0] * coeff_mod_size
    for i in range(coeff_mod_size):
        result[i] = [0] * coeff_count

    for j in range(coeff_mod_size):
        modulus = coeff_modulus[j]
        max_multiple = max_random - (max_random % modulus) - 1
        for i in range(coeff_count):
            # This ensures uniform distribution.
            while True:
                rand = randbits(32) << 31 | randbits(32) >> 1
                if rand < max_multiple:
                    break
            result[j][i] = rand % modulus
    return result


def encrypt_asymmetric(context, public_key):
    """Create encryption of zero values with a public key which can be used in subsequent processes to add a message into it.

    Args:
        context: A valid EncryptionParam class object required for extracting the encryption parameters.
        public_key: A PublicKey object generated with same encryption parameters.
    """
    param = context.param
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus_degree
    encrypted_size = 2

    # Generate u <-- R_3
    u = sample_poly_ternary(param)

    c_0 = [0] * coeff_mod_size
    for i in range(coeff_mod_size):
        c_0[i] = [0] * coeff_count

    c_1 = [0] * coeff_mod_size
    for i in range(coeff_mod_size):
        c_1[i] = [0] * coeff_count
    result = [c_0, c_1]

    # c[j] = u * public_key[j]
    # Generate e_j <-- chi
    # c[j] = public_key[j] * u + e[j]
    for k in range(encrypted_size):
        e = sample_poly_normal(param)
        for j in range(coeff_mod_size):
            for i in range(coeff_count):
                result[k][j][i] = add_mod(
                    multiply_mod(u[j][i], public_key[k][j][i], coeff_modulus[j]),
                    e[j][i],
                    coeff_modulus[j],
                )

    return CipherText(result)  # result = public_key[j] * u + e[j]


def encrypt_symmetric(context, secret_key):
    """Create encryption of zero values with a secret key which can be used in subsequent processes to add a message into it.

    Args:
        context: A valid EncryptionParam class object required for extracting the encryption parameters.
        secret_key: A SecretKey object generated with same encryption parameters.
    """
    coeff_modulus = context.param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = context.param.poly_modulus_degree

    # Sample uniformly at random
    c1 = sample_poly_uniform(context.param)

    # Sample e <-- chi
    e = sample_poly_normal(context.param)

    # calculate -(a*s + e) (mod q) and store in c0

    c0 = [0] * coeff_mod_size
    for i in range(coeff_mod_size):
        c0[i] = [0] * coeff_count  # c0 = [[0] * coeff_count] * coeff_mod_size

    for j in range(coeff_mod_size):
        for i in range(coeff_count):
            c0[j][i] = negate_mod(
                add_mod(
                    multiply_mod(c1[j][i], secret_key[j][i], coeff_modulus[j]),
                    e[j][i],
                    coeff_modulus[j],
                ),
                coeff_modulus[j],
            )

    return CipherText([c0, c1])
