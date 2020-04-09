from tenseal import context, ckks_vector, SCHEME_TYPE
from _tenseal_cpp import CKKSVector


DEFAULT_CKKS_N = 8192
DEFAULT_CKKS_COEFF_MOD = [60, 40, 40, 60]
DEFAULT_CKKS_SCALE = 2 ** 40


def generate_ckks_keys(poly_modulus_degree=DEFAULT_CKKS_N,
                        coeff_mod_bit_sizes=DEFAULT_CKKS_COEFF_MOD):
    """Returns a public context (containing public keys and attributes)
    and the secret key.
    """
    c = context(SCHEME_TYPE.CKKS, poly_modulus_degree, 0, coeff_mod_bit_sizes)
    secret_key = c.secret_key()
    c.make_context_public()

    return c, secret_key
