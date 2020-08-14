""" Largest allowed bit counts for coeff_modulus based on the security estimates from
    HomomorphicEncryption.org security standard. Scheme samples the secret key
    from a ternary {-1, 0, 1} distribution."""


# Ternary secret: 128 bits classical security
def FV_STD_PARMS_128_TC(poly_modulus_degree):
    _dict = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}
    return _dict.get(poly_modulus_degree, 0)


# Ternary secret: 192 bits classical security
def FV_STD_PARMS_192_TC(poly_modulus_degree):
    _dict = {1024: 19, 2048: 37, 4096: 75, 8192: 152, 16384: 305, 32768: 611}
    return _dict.get(poly_modulus_degree, 0)


# Ternary secret; 256 bits classical security
def FV_STD_PARMS_256_TC(poly_modulus_degree):
    _dict = {1024: 14, 2048: 29, 4096: 58, 8192: 118, 16384: 237, 32768: 476}
    return _dict.get(poly_modulus_degree, 0)
