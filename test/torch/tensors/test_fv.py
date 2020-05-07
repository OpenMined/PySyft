import pytest

from syft.frameworks.torch.he.fv.util.numth import is_prime
from syft.frameworks.torch.he.fv.util.operations import multiply_many_except
from syft.frameworks.torch.he.fv.modulus import CoeffModulus
from syft.frameworks.torch.he.fv.encryption_params import EncryptionParams
from syft.frameworks.torch.he.fv.modulus import SeqLevelType
from syft.frameworks.torch.he.fv.context import Context
from syft.frameworks.torch.he.fv.util.operations import get_significant_count
from syft.frameworks.torch.he.fv.integer_encoder import IntegerEncoder
from syft.frameworks.torch.he.fv.key_generator import KeyGenerator
from syft.frameworks.torch.he.fv.util.base_converter import BaseConvertor
from syft.frameworks.torch.he.fv.util.rns_base import RNSBase
from syft.frameworks.torch.he.fv.util.operations import try_invert_int_mod
from syft.frameworks.torch.he.fv.util.operations import xgcd


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


@pytest.mark.parametrize(
    "poly_modulus_degree, plain_modulus, coeff_bit_sizes",
    [(1, 2, [30, 40, 50]), (-64, 64, [30, 60, 60])],
)
def test_EncryptionParams_exceptions(poly_modulus_degree, plain_modulus, coeff_bit_sizes):
    with pytest.raises(ValueError):
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
        (0xFFFFFFFFFFFFFFF, -3),
        (0xFFFFFFFFFFFFFFF, 64),
        (0xFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
        (0xFFFFFFFFFFFFFFF, 0x80F02),
        (1024, 64),
    ],
)
def test_integer_encoder(plain_modulus, value):
    # TODO For creating IntegerEncoder we should not need to create coeff_modulus so need to change the way we pass ctx object in IntegerEncoder class.
    # due to which currently we have to create coeff modulus of very big size and it slows down these tests.
    enc_param = EncryptionParams()
    enc_param.plain_modulus = plain_modulus
    enc_param.coeff_modulus = CoeffModulus().create(plain_modulus, [100, 100, 100])
    ctx = Context(enc_param)
    encoder = IntegerEncoder(ctx)
    poly = encoder.encode(value)
    assert value == encoder.decode(poly)


@pytest.mark.parametrize(
    "operand, count, exp, result",
    [
        ([0, 0, 0], 2, 0, 0),
        ([0, 0, 0], 3, 0, 0),
        ([0, 0, 0], 2, 0, 0),
        ([2, 3, 5], 2, 0, 3),
        ([2, 3, 5], 2, 1, 2),
        ([2, 3, 5], 3, 0, 15),
        ([2, 3, 5], 3, 1, 10),
        ([2, 3, 5], 3, 2, 6),
    ],
)
def test_multiply_many_except(operand, count, exp, result):
    assert multiply_many_except(operand, count, exp) == result


@pytest.mark.parametrize(
    "x, y, result",
    [
        (7, 7, [7, 0, 1]),
        (2, 2, [2, 0, 1]),
        (1, 1, [1, 0, 1]),
        (1, 2, [1, 1, 0]),
        (5, 6, [1, -1, 1]),
        (13, 19, [1, 3, -2]),
        (14, 21, [7, -1, 1]),
        (2, 1, [1, 0, 1]),
        (6, 5, [1, 1, -1]),
        (19, 13, [1, -2, 3]),
        (21, 14, [7, 1, -1]),
    ],
)
def test_xgcd(x, y, result):
    assert result == xgcd(x, y)


@pytest.mark.parametrize(
    "input, modulus, result", [(1, 2, 1), (3, 2, 1), (0xFFFFFF, 2, 1), (5, 19, 4), (4, 19, 5)]
)
def test_try_invert_int_mod(input, modulus, result):
    assert result == try_invert_int_mod(input, modulus)


@pytest.mark.parametrize(
    "ibase, obase, input, output",
    [
        ([3], [2], [0, 1, 2], [0, 1, 0]),
        ([2, 3], [2], [0, 1, 0, 0, 1, 2], [0, 1, 0]),
        ([2, 3], [2, 3], [1, 1, 0, 1, 2, 2], [1, 1, 0, 1, 2, 2]),
        ([2, 3], [3, 4, 5], [0, 1, 1, 0, 1, 2], [0, 1, 2, 0, 3, 1, 0, 2, 0]),
    ],
)
def test_fast_convert_array(ibase, obase, input, output):
    base_converter = BaseConvertor(RNSBase(ibase), RNSBase(obase))
    result = base_converter.fast_convert_array(input, 3)
    for i in range(len(result)):
        assert result[i] == output[i]
