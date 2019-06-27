import sys
import pytest


@pytest.mark.usefixtures("no_tf_encrypted")
def test_tf_encrypted_missing_import_syft():

    sys.modules.pop("syft", None)
    sys.modules.pop("syft.dependency_check", None)
    sys.modules.pop("tf_encrypted", None)
    sys.modules.pop("tfe", None)
    import syft

    with pytest.raises(AttributeError):
        kerasHook = syft.KerasHook

    with pytest.raises(AttributeError):
        tfeWorker = syft.TFEWorker
