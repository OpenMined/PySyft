import pytest

from syft.frameworks.torch.he.fv.modulus import CoeffModulus
from syft.frameworks.torch.he.fv.modulus import SeqLevelType


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


def test_CoeffModulus_bfv_default():
    coeffModulus = CoeffModulus()
    assert len(coeffModulus.bfv_default(1024, SeqLevelType.TC128)) == 1
    assert len(coeffModulus.bfv_default(1024, SeqLevelType.TC192)) == 1
    assert len(coeffModulus.bfv_default(1024, SeqLevelType.TC256)) == 1
    assert len(coeffModulus.bfv_default(1024)) == 1

    assert len(coeffModulus.bfv_default(2048, SeqLevelType.TC128)) == 1
    assert len(coeffModulus.bfv_default(2048, SeqLevelType.TC192)) == 1
    assert len(coeffModulus.bfv_default(2048, SeqLevelType.TC256)) == 1
    assert len(coeffModulus.bfv_default(2048)) == 1

    assert len(coeffModulus.bfv_default(4096, SeqLevelType.TC128)) == 3
    assert len(coeffModulus.bfv_default(4096, SeqLevelType.TC192)) == 3
    assert len(coeffModulus.bfv_default(4096, SeqLevelType.TC256)) == 1
    assert len(coeffModulus.bfv_default(4096)) == 3

    assert len(coeffModulus.bfv_default(8192, SeqLevelType.TC128)) == 5
    assert len(coeffModulus.bfv_default(8192, SeqLevelType.TC192)) == 4
    assert len(coeffModulus.bfv_default(8192, SeqLevelType.TC256)) == 3
    assert len(coeffModulus.bfv_default(8192)) == 5

    assert len(coeffModulus.bfv_default(16384, SeqLevelType.TC128)) == 9
    assert len(coeffModulus.bfv_default(16384, SeqLevelType.TC192)) == 6
    assert len(coeffModulus.bfv_default(16384, SeqLevelType.TC256)) == 5
    assert len(coeffModulus.bfv_default(16384)) == 9

    assert len(coeffModulus.bfv_default(32768, SeqLevelType.TC128)) == 16
    assert len(coeffModulus.bfv_default(32768, SeqLevelType.TC192)) == 11
    assert len(coeffModulus.bfv_default(32768, SeqLevelType.TC256)) == 9
    assert len(coeffModulus.bfv_default(32768)) == 16
