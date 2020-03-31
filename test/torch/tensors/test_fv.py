import pytest

from syft.frameworks.torch.he.fv.util.numth import is_prime
from syft.frameworks.torch.he.fv.modulus import CoeffModulus
from syft.frameworks.torch.he.fv.encryption_params import EncryptionParams
from syft.frameworks.torch.he.fv.modulus import SeqLevelType
from syft.frameworks.torch.he.fv.context import Context
from syft.frameworks.torch.he.fv.util.operations import get_significant_count
from syft.frameworks.torch.he.fv.integer_encoder import IntegerEncoder


@pytest.mark.parametrize(
    "num, status",
    [
        (0, False),
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (221, False),
        (65537, True),
        (65536, False),
        (59399, True),
        (72307, True),
        (36893488147419103, True),
        (36893488147419107, False),
        (72307 * 59399, False),
    ],
)
def test_is_prime(num, status):
    assert is_prime(num) == status


@pytest.mark.parametrize(
    "poly_modulus_degree, plain_modulus, coeff_bit_sizes",
    [(128, 2, [30, 40, 50]), (1024, 64, [30, 60, 60]), (64, 64, [30])],
)
def test_EncryptionParams(poly_modulus_degree, plain_modulus, coeff_bit_sizes):
    params = EncryptionParams()
    params.poly_modulus_degree = poly_modulus_degree
    params.plain_modulus = plain_modulus
    cm = CoeffModulus()
    params.coeff_modulus = cm.create(poly_modulus_degree, coeff_bit_sizes)

    for i in range(len(coeff_bit_sizes)):
        assert is_prime(params.coeff_modulus[i])


def test_CoeffModulus_create():
    coeffModulus = CoeffModulus()
    assert len(coeffModulus.create(2, [])) == 0

    cm = coeffModulus.create(2, [3])
    assert len(cm) == 1
    assert cm[0] == 5

    cm = coeffModulus.create(2, [3, 4])
    assert len(cm) == 2
    assert cm[0] == 5
    assert cm[1] == 13

    cm = coeffModulus.create(2, [3, 5, 4, 5])
    assert len(cm) == 4
    assert cm[0] == 5
    assert cm[1] == 17
    assert cm[2] == 13
    assert cm[3] == 29

    cm = coeffModulus.create(32, [30, 40, 30, 30, 40])
    assert len(cm) == 5
    assert cm[0] % 64 == 1
    assert cm[1] % 64 == 1
    assert cm[2] % 64 == 1
    assert cm[3] % 64 == 1
    assert cm[4] % 64 == 1


@pytest.mark.parametrize(
    "poly_modulus_degree, SeqLevelType, result",
    [
        (1024, SeqLevelType.TC128, 1),
        (1024, SeqLevelType.TC192, 1),
        (1024, SeqLevelType.TC256, 1),
        (2048, SeqLevelType.TC128, 1),
        (2048, SeqLevelType.TC192, 1),
        (2048, SeqLevelType.TC256, 1),
        (4096, SeqLevelType.TC128, 3),
        (4096, SeqLevelType.TC192, 3),
        (4096, SeqLevelType.TC256, 1),
        (8192, SeqLevelType.TC128, 5),
        (8192, SeqLevelType.TC192, 4),
        (8192, SeqLevelType.TC256, 3),
        (16384, SeqLevelType.TC128, 9),
        (16384, SeqLevelType.TC192, 6),
        (16384, SeqLevelType.TC256, 5),
        (32768, SeqLevelType.TC128, 16),
        (32768, SeqLevelType.TC192, 11),
        (32768, SeqLevelType.TC256, 9),
    ],
)
def test_CoeffModulus_bfv_default(poly_modulus_degree, SeqLevelType, result):
    coeffModulus = CoeffModulus()
    assert len(coeffModulus.bfv_default(poly_modulus_degree, SeqLevelType)) == result


@pytest.mark.parametrize(
    "ptr, result",
    [
        ([0, 0], 0),
        ([1, 0], 1),
        ([2, 0], 1),
        ([0xFFFFFFFFFFFFFFFF, 0], 1),
        ([0, 1], 2),
        ([0xFFFFFFFFFFFFFFFF, 1], 2),
        ([0xFFFFFFFFFFFFFFFF, 0x8000000000000000], 2),
        ([0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF], 2),
    ],
)
def test_get_significant_count(ptr, result):
    assert result == get_significant_count(ptr, 2)


@pytest.mark.parametrize(
    "plain_modulus, value",
    [
        (0xFFFFFFFFFFFFFFF, 1),
        (0xFFFFFFFFFFFFFFF, 2),
        (0xFFFFFFFFFFFFFFF, 3),
        (0xFFFFFFFFFFFFFFF, 64),
        (0xFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
        (0xFFFFFFFFFFFFFFF, 0x80F02),
        (1024, 64),
    ],
)
def test_integer_encoder(plain_modulus, value):
    enc_param = EncryptionParams()
    enc_param.plain_modulus = plain_modulus
    ctx = Context(enc_param)
    encoder = IntegerEncoder(ctx)

    poly = encoder.encode(value)
    assert value == encoder.decode(poly)
