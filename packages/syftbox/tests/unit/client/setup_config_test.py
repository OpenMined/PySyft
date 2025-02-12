from secrets import token_hex

from syftbox.client.cli_setup import setup_config_interactive
from syftbox.lib.client_config import SyftClientConfig


def test_setup_new_config(tmp_path):
    config_path = tmp_path / f"config_{token_hex}.json"
    data_dir = tmp_path / "data"
    email = "test@example.com"
    server = "http://test.com/"
    port = 8080

    result = setup_config_interactive(
        config_path=config_path,
        email=email,
        data_dir=data_dir,
        server=server,
        port=port,
        skip_auth=True,
        skip_verify_install=True,
    )

    assert isinstance(result, SyftClientConfig)
    assert result.path == config_path
    assert result.data_dir == data_dir
    assert str(result.email) == str(email)
    assert str(result.server_url) == str(server)
    assert str(result.client_url) == "http://127.0.0.1:8080/"
    assert result.client_url.port == port


def test_setup_new_config_with_prompt(tmp_path, monkeypatch):
    config_path = tmp_path / f"config_{token_hex}.json"
    data_dir = tmp_path / "data"
    email = "test@example.com"
    server = "http://test.com/"
    port = 8080

    monkeypatch.setattr("syftbox.client.cli_setup.prompt_email", lambda: email)
    monkeypatch.setattr("syftbox.client.cli_setup.prompt_data_dir", lambda: data_dir)

    result = setup_config_interactive(
        config_path=config_path,
        email=None,
        data_dir=None,
        server=server,
        port=port,
        skip_auth=True,
        skip_verify_install=True,
    )

    assert isinstance(result, SyftClientConfig)
    assert result.path == config_path
    assert result.data_dir == data_dir
    assert str(result.email) == str(email)
    assert str(result.server_url) == str(server)
    assert str(result.client_url) == "http://127.0.0.1:8080/"
    assert result.client_url.port == port


def test_setup_existing_config(tmp_path, mock_config):
    new_port = 8081
    result = setup_config_interactive(
        server=str(mock_config.server_url),
        email=str(mock_config.email),
        data_dir="",
        config_path=mock_config.path,
        port=new_port,
        skip_auth=True,
        skip_verify_install=True,
    )

    assert isinstance(result, SyftClientConfig)
    assert result.client_url.port == new_port
