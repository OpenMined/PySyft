import sys
import pytest


def test_tf_encrypted_available():
    sys.modules.pop("syft", None)
    sys.modules.pop("syft.dependency_check", None)
    from syft import dependency_check

    assert dependency_check.keras_available


@pytest.mark.usefixtures("no_tf_encrypted")
def test_tf_encrypted_missing():

    sys.modules.pop("syft.dependency_check", None)
    sys.modules.pop("tf_encrypted", None)
    sys.modules.pop("tfe", None)
    import syft.dependency_check

    assert not syft.dependency_check.keras_available
