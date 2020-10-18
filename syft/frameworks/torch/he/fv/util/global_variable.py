"""Attributes:
    DEFAULT_C0EFF_MODULUS_128: A dictionary that maps degrees of the polynomial
    modulus to list of Modulus elements so that when used with the default value
    for the standard deviation of the noise distribution (noise_standard_deviation),
    the security level is at least 128 bits according to http://HomomorphicEncryption.org.
    This makes it easy for non-expert users to select secure parameters.

    DEFAULT_C0EFF_MODULUS_192: A dictionary that maps degrees of the polynomial
    modulus to list of Modulus elements so that when used with the default value
    for the standard deviation of the noise distribution (noise_standard_deviation),
    the security level is at least 192 bits according to http://HomomorphicEncryption.org.
    This makes it easy for non-expert users to select secure parameters.

    DEFAULT_C0EFF_MODULUS_256: A dictionary that maps degrees of the polynomial
    modulus to list of Modulus elements so that when used with the default value
    for the standard deviation of the noise distribution (noise_standard_deviation),
    the security level is at least 256 bits according to http://HomomorphicEncryption.org.
    This makes it easy for non-expert users to select secure parameters.
"""
DEFAULT_C0EFF_MODULUS_128 = {
    # Polynomial modulus: 1x^1024 + 1
    # Modulus count: 1
    # Total bit count: 27
    1024: [0x7E00001],
    # Polynomial modulus: 1x^2048 + 1
    # Modulus count: 1
    # Total bit count: 54
    2048: [0x3FFFFFFF000001],
    # Polynomial modulus: 1x^4096 + 1
    # Modulus count: 3
    # Total bit count: 109 = 2 * 36 + 37
    4096: [0xFFFFEE001, 0xFFFFC4001, 0x1FFFFE0001],
    # Polynomial modulus: 1x^8192 + 1
    # Modulus count: 5
    # Total bit count: 218 = 2 * 43 + 3 * 44
    8192: [0x7FFFFFD8001, 0x7FFFFFC8001, 0xFFFFFFFC001, 0xFFFFFF6C001, 0xFFFFFEBC001],
    # Polynomial modulus: 1x^16384 + 1
    # Modulus count: 9
    # Total bit count: 438 = 3 * 48 + 6 * 49
    16384: [
        0xFFFFFFFD8001,
        0xFFFFFFFA0001,
        0xFFFFFFF00001,
        0x1FFFFFFF68001,
        0x1FFFFFFF50001,
        0x1FFFFFFEE8001,
        0x1FFFFFFEA0001,
        0x1FFFFFFE88001,
        0x1FFFFFFE48001,
    ],
    # Polynomial modulus: 1x^32768 + 1
    # Modulus count: 16
    # Total bit count: 881 = 15 * 55 + 56
    32768: [
        0x7FFFFFFFE90001,
        0x7FFFFFFFBF0001,
        0x7FFFFFFFBD0001,
        0x7FFFFFFFBA0001,
        0x7FFFFFFFAA0001,
        0x7FFFFFFFA50001,
        0x7FFFFFFF9F0001,
        0x7FFFFFFF7E0001,
        0x7FFFFFFF770001,
        0x7FFFFFFF380001,
        0x7FFFFFFF330001,
        0x7FFFFFFF2D0001,
        0x7FFFFFFF170001,
        0x7FFFFFFF150001,
        0x7FFFFFFEF00001,
        0xFFFFFFFFF70001,
    ],
}

DEFAULT_C0EFF_MODULUS_192 = {
    # Polynomial modulus: 1x^1024 + 1
    # Modulus count: 1
    # Total bit count: 19
    1024: [0x7F001],
    # Polynomial modulus: 1x^2048 + 1
    # Modulus count: 1
    # Total bit count: 37
    2048: [0x1FFFFC0001],
    # Polynomial modulus: 1x^4096 + 1
    # Modulus count: 3
    # Total bit count: 75 = 3 * 25
    4096: [0x1FFC001, 0x1FCE001, 0x1FC0001],
    # Polynomial modulus: 1x^8192 + 1
    # Modulus count: 4
    # Total bit count: 152 = 4 * 38
    8192: [0x3FFFFAC001, 0x3FFFF54001, 0x3FFFF48001, 0x3FFFF28001],
    # Polynomial modulus: 1x^16384 + 1
    # Modulus count: 6
    # Total bit count: 300 = 6 * 50
    16384: [
        0x3FFFFFFDF0001,
        0x3FFFFFFD48001,
        0x3FFFFFFD20001,
        0x3FFFFFFD18001,
        0x3FFFFFFCD0001,
        0x3FFFFFFC70001,
    ],
    # Polynomial modulus: 1x^32768 + 1
    # Modulus count: 11
    # Total bit count: 600 = 5 * 54 + 6 * 55
    32768: [
        0x3FFFFFFFD60001,
        0x3FFFFFFFCA0001,
        0x3FFFFFFF6D0001,
        0x3FFFFFFF5D0001,
        0x3FFFFFFF550001,
        0x7FFFFFFFE90001,
        0x7FFFFFFFBF0001,
        0x7FFFFFFFBD0001,
        0x7FFFFFFFBA0001,
        0x7FFFFFFFAA0001,
        0x7FFFFFFFA50001,
    ],
}

DEFAULT_C0EFF_MODULUS_256 = {
    # Polynomial modulus: 1x^1024 + 1
    # Modulus count: 1
    # Total bit count: 14
    1024: [0x3001],
    # Polynomial modulus: 1x^2048 + 1
    # Modulus count: 1
    # Total bit count: 29
    2048: [0x1FFC0001],
    # Polynomial modulus: 1x^4096 + 1
    # Modulus count: 1
    # Total bit count: 58
    4096: [0x3FFFFFFFF040001],
    # Polynomial modulus: 1x^8192 + 1
    # Modulus count: 3
    # Total bit count: 118 = 2 * 39 + 40
    8192: [0x7FFFFEC001, 0x7FFFFB0001, 0xFFFFFDC001],
    # Polynomial modulus: 1x^16384 + 1
    # Modulus count: 5
    # Total bit count: 237 = 3 * 47 + 2 * 48
    16384: [0x7FFFFFFC8001, 0x7FFFFFF00001, 0x7FFFFFE70001, 0xFFFFFFFD8001, 0xFFFFFFFA0001],
    # Polynomial modulus: 1x^32768 + 1
    # Modulus count: 9
    # Total bit count: 476 = 52 + 8 * 53
    32768: [
        0xFFFFFFFF00001,
        0x1FFFFFFFE30001,
        0x1FFFFFFFD80001,
        0x1FFFFFFFD10001,
        0x1FFFFFFFC50001,
        0x1FFFFFFFBF0001,
        0x1FFFFFFFB90001,
        0x1FFFFFFFB60001,
        0x1FFFFFFFA50001,
    ],
}

gamma = 0x1FFFFFFFFFC80001


COEFF_MOD_COUNT_MIN = 1
COEFF_MOD_COUNT_MAX = 62

COEFF_MOD_BIT_COUNT_MIN = 2
COEFF_MOD_BIT_COUNT_MAX = 60

POLY_MOD_DEGREE_MIN = 2
POLY_MOD_DEGREE_MAX = 32768

NOISE_STANDARD_DEVIATION = 3.20
NOISE_DISTRIBUTION_WIDTH_MULTIPLIER = 6
NOISE_MAX_DEVIATION = NOISE_DISTRIBUTION_WIDTH_MULTIPLIER * NOISE_STANDARD_DEVIATION
