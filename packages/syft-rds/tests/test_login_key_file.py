from pathlib import Path
from unittest.mock import MagicMock

import pytest

import syft.sync.login
from syft.sync.environments.environment import Environment
from syft.sync.peers.peer_store import PeerStore
from syft_rds import login as rds_login

EMAIL = "do@openmined.org"


@pytest.mark.parametrize("login", [rds_login.login_do, rds_login.login_ds])
def test_login_restores_uploaded_key_file(tmp_path, monkeypatch, login):
    uploaded = tmp_path / "upload" / "crypto_keys.json"
    earlier = PeerStore.create(email=EMAIL, keys_path=uploaded)

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    for module in (rds_login, syft.sync.login):
        monkeypatch.setattr(module, "check_env", lambda: Environment.JUPYTER)
    monkeypatch.setattr(
        rds_login,
        "handle_potential_version_mismatches_on_login",
        lambda email, token_path: False,
    )
    monkeypatch.setattr(rds_login, "_init_client_login", lambda *a, **kw: None)
    configs = []
    monkeypatch.setattr(
        rds_login.SyftRDSClient,
        "from_config",
        classmethod(lambda cls, config: configs.append(config) or MagicMock()),
    )

    login(EMAIL, token_path=tmp_path / "token.json", key_file=uploaded)

    keys_path = configs[0].sync.peer_manager_config.crypto_keys_path
    restored = PeerStore.create(email=EMAIL, keys_path=keys_path)
    assert restored.get_public_bundle() == earlier.get_public_bundle()
