import torch as th
from secrets import SystemRandom
from secrets import randbits
from torch.distributions import Normal

from syft.frameworks.torch.he.fv.ciphertext import CipherText
from syft.frameworks.torch.he.fv.util.operations import poly_add_mod
from syft.frameworks.torch.he.fv.util.operations import poly_mul_mod
from syft.frameworks.torch.he.fv.util.operations import poly_negate_mod
from syft.frameworks.torch.he.fv.util.global_variable import NOISE_STANDARD_DEVIATION


def sample_poly_ternary(parms):
    """Generate a ternary polynomial uniformally with elements [-1, 0, 1]
    where -1 is represented as (modulus - 1) because -1 % modulus == modulus - 1.

    Used for generating secret key using coeff_modulus(list of prime nos) which
    represents as 'q' in the research paper.

    Args:
       parms (EncryptionParam): Encryption parameters.

    Returns:
        A 2-dim list having integer from [-1, 0, 1].
    """
    coeff_modulus = parms.coeff_modulus
    coeff_count = parms.poly_modulus
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
    """Generate a polynomial from normal distribution where negative values are
    represented by (neg_val % modulus) a positive value.

    Args:
        parms (EncryptionParam): Encryption parameters.

    Returns:
        A 2-dim list having integer from normal distributions.
    """
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus

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
        parms (EncryptionParam): Encryption parameters.
    Returns:
        A 2-dim list having integer from uniform distributions.
    """
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    coeff_count = param.poly_modulus

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


def encrypt_asymmetric(context, param_id, public_key):
    """Create encryption of zero values with a public key which can be used in
    subsequent processes to add a message into it.

    Args:
        context (Context): A valid context required for extracting the encryption
            parameters.
        param_id: Parameter id for accessing the correct parameters from the context chain.
        public_key (PublicKey): A public key generated with same encryption parameters.

    Returns:
        A ciphertext object containing encryption of zeroes by asymmetric encryption procedure.
    """
    param = context.context_data_map[param_id].param
    poly_mod = param.poly_modulus
    coeff_modulus = param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)
    encrypted_size = len(public_key)

    # Generate u <-- R_3
    u = sample_poly_ternary(param)

    c_0 = [0] * coeff_mod_size

    c_1 = [0] * coeff_mod_size
    result = [c_0, c_1]

    # c[i] = u * public_key[i]
    # Generate e_j <-- chi
    # c[i] = public_key[i] * u + e[i]
    for j in range(encrypted_size):
        e = sample_poly_normal(param)
        for i in range(coeff_mod_size):
            result[j][i] = poly_add_mod(
                poly_mul_mod(public_key[j][i], u[i], coeff_modulus[i], poly_mod),
                e[i],
                coeff_modulus[i],
                poly_mod,
            )
    return CipherText(result, param_id)


def encrypt_symmetric(context, param_id, secret_key):
    """Create encryption of zero values with a secret key which can be used in subsequent
    processes to add a message into it.

    Args:
        context (Context): A valid context required for extracting the encryption parameters.
        param_id: Parameter id for accessing the correct parameters from the context chain.
        secret_key (SecretKey): A secret key generated with same encryption parameters.

    Returns:
        A ciphertext object containing encryption of zeroes by symmetric encryption procedure.
    """
    key_param = context.context_data_map[param_id].param
    poly_mod = key_param.poly_modulus
    coeff_modulus = key_param.coeff_modulus
    coeff_mod_size = len(coeff_modulus)

    # Sample uniformly at random
    c1 = sample_poly_uniform(key_param)

    # Sample e <-- chi
    e = sample_poly_normal(key_param)

    # calculate -(a*s + e) (mod q) and store in c0

    c0 = [0] * coeff_mod_size

    for i in range(coeff_mod_size):
        c0[i] = poly_negate_mod(
            poly_add_mod(
                poly_mul_mod(c1[i], secret_key[i], coeff_modulus[i], poly_mod),
                e[i],
                coeff_modulus[i],
                poly_mod,
            ),
            coeff_modulus[i],
        )

    return CipherText([c0, c1], param_id)
