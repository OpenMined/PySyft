# third party
import pytest

# syft absolute
from syft.__init__ import __version__
from syft.service.metadata.server_metadata import check_version


def test_check_base_version_success() -> None:
    response = check_version(
        client_version=__version__, server_version=__version__, server_name="test"
    )

    assert response is True


def test_check_base_version_fail() -> None:
    with pytest.raises(ValueError):
        check_version(
            client_version="x.x.x", server_version=__version__, server_name="test"
        )


def test_check_pre_version_fail() -> None:
    response = check_version(
        client_version="0.8.0-beta.0",
        server_version="0.8.0-beta.1",
        server_name="test",
        silent=False,
    )
    assert response is False
