# syft absolute
from syft.lib.python.string import String

STRING_PYTHON = "Hello OpenMined"
STRING_SYFT = String(STRING_PYTHON)


def test_id_concat_python_type() -> None:
    val = " 2020"

    result = STRING_SYFT + val

    assert result.id
    assert result.id != STRING_SYFT.id
    assert result == STRING_PYTHON + val

    result = val + STRING_SYFT

    assert result.id
    assert result.id != STRING_SYFT.id
    assert val + STRING_PYTHON == result


def test_id_concat_syft_type() -> None:
    val = String(" 2020")

    result = STRING_SYFT + val

    assert result.id
    assert result.id != STRING_SYFT.id
    assert val.id != result.id
    assert result == STRING_PYTHON + val

    result = val + STRING_SYFT

    assert result.id
    assert result.id != STRING_SYFT.id
    assert val.id != result.id
    assert val + STRING_PYTHON == result
