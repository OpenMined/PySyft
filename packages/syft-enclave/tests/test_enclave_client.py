import os


os.environ["PRE_SYNC"] = "false"

from syft_enclaves import SyftEnclaveClient


def test_quad_initialization():
    enclave, do1, do2, ds = SyftEnclaveClient.quad_with_mock_drive_service_connection()

    # 4 clients returned
    clients = [enclave, do1, do2, ds]
    assert len(clients) == 4
    assert all(isinstance(c, SyftEnclaveClient) for c in clients)

    # Correct roles
    assert enclave._rds.has_do_role is True
    assert enclave._rds.has_ds_role is True

    assert do1._rds.has_do_role is True
    assert do1._rds.has_ds_role is True

    assert do2._rds.has_do_role is True
    assert do2._rds.has_ds_role is True

    assert ds._rds.has_do_role is False
    assert ds._rds.has_ds_role is True

    # Helper to get approved peer emails for a client
    def approved_emails(client):
        return {p.email for p in client._rds.peer_manager.approved_peers}

    # Enclave (DO-only): approved DS, DO1, DO2
    assert approved_emails(enclave) == {ds.email, do1.email, do2.email}

    # DO1 (dual): approved DS and enclave as DO
    assert approved_emails(do1) == {ds.email, enclave.email}

    # DO2 (dual): approved DS and enclave as DO
    assert approved_emails(do2) == {ds.email, enclave.email}

    # DS: all peers are accepted (both sides created folders)
    assert approved_emails(ds) == {do1.email, do2.email, enclave.email}
