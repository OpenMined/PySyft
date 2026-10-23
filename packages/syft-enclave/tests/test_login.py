from pathlib import Path
from unittest.mock import MagicMock

import syft.sync.login
from syft.sync.environments.environment import Environment
from syft.sync.peers.peer_store import PeerStore
from syft_enclaves import login as enclave_login

EMAIL = "do@openmined.org"


def test_login_restores_uploaded_key_file(tmp_path, monkeypatch):
    uploaded = tmp_path / "upload" / "crypto_keys.json"
    earlier = PeerStore.create(email=EMAIL, keys_path=uploaded)

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    for module in (enclave_login, syft.sync.login):
        monkeypatch.setattr(module, "check_env", lambda: Environment.JUPYTER)
    monkeypatch.setattr(
        enclave_login,
        "handle_potential_version_mismatches_on_login",
        lambda email, token_path: False,
    )
    monkeypatch.setattr(enclave_login, "_init_client_login", lambda *a, **kw: None)
    configs = []
    monkeypatch.setattr(
        enclave_login.SyftEnclaveClient,
        "from_config",
        classmethod(lambda cls, config: configs.append(config) or MagicMock()),
    )

    enclave_login.login_do(EMAIL, token_path=tmp_path / "token.json", key_file=uploaded)

    keys_path = configs[0].sync.peer_manager_config.crypto_keys_path
    restored = PeerStore.create(email=EMAIL, keys_path=keys_path)
    assert restored.get_public_bundle() == earlier.get_public_bundle()
