import pytest

from syft_client.sync.login import _verify_token_matches_email
from syft_client.sync.syftbox_manager import SyftboxManager


def test_get_authenticated_email_matches_constructor_email():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()

    assert ds_manager._connection_router.get_authenticated_email() == ds_manager.email
    assert do_manager._connection_router.get_authenticated_email() == do_manager.email


def test_verify_token_matches_email_passes_when_matching():
    ds_manager, _ = SyftboxManager.pair_with_mock_drive_service_connection()
    _verify_token_matches_email(ds_manager)


def _drive_service(manager):
    return manager._connection_router.connections[0].drive_service


def test_verify_token_matches_email_raises_on_mismatch():
    ds_manager, _ = SyftboxManager.pair_with_mock_drive_service_connection()

    _drive_service(ds_manager)._current_user = "someone-else@gmail.com"

    with pytest.raises(ValueError, match="Token/email mismatch") as exc_info:
        _verify_token_matches_email(ds_manager)

    msg = str(exc_info.value)
    assert ds_manager.email in msg
    assert "someone-else@gmail.com" in msg


def test_verify_token_matches_email_is_case_insensitive():
    ds_manager, _ = SyftboxManager.pair_with_mock_drive_service_connection()

    _drive_service(ds_manager)._current_user = ds_manager.email.upper()

    _verify_token_matches_email(ds_manager)
