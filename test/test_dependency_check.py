import sys
import pytest
from syft import dependency_check


@pytest.mark.skipif(not dependency_check.tensorflow_available, reason="tf 2.0+ not installed")
def test_tensorflow_available():  # pragma: no cover
    sys.modules.pop("syft", None)
    sys.modules.pop("syft.dependency_check", None)
    from syft import dependency_check

    assert dependency_check.tensorflow_available


@pytest.mark.skipif(not dependency_check.tfe_available, reason="tf_encrypted not installed")
def test_tf_encrypted_available():  # pragma: no cover
    sys.modules.pop("syft", None)
    sys.modules.pop("syft.dependency_check", None)
    from syft import dependency_check

    assert dependency_check.tfe_available


@pytest.mark.skipif(not dependency_check.torch_available, reason="torch not installed")
def test_torch_available():  # pragma: no cover
    sys.modules.pop("syft", None)
    sys.modules.pop("syft.dependency_check", None)
    from syft import dependency_check

    assert dependency_check.torch_available


@pytest.mark.usefixtures("hide_module")
def test_tensorflow_missing():  # pragma: no cover
    sys.modules.pop("syft", None)
    sys.modules.pop("syft.dependency_check", None)
    sys.modules.pop("tensorflow", None)
    sys.modules.pop("tf", None)
    import syft.dependency_check

    assert not syft.dependency_check.tensorflow_available


@pytest.mark.usefixtures("hide_module")
def test_tf_encrypted_missing():  # pragma: no cover
    sys.modules.pop("syft.dependency_check", None)
    sys.modules.pop("tf_encrypted", None)
    sys.modules.pop("tfe", None)
    import syft.dependency_check

    assert not syft.dependency_check.tfe_available
