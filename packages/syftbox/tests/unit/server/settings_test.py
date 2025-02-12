import os
from pathlib import Path

from syftbox.server.settings import ServerSettings


def test_server_settings_from_env():
    os.environ["SYFTBOX_DATA_FOLDER"] = "data_folder"

    settings = ServerSettings()
    # must be absolute!
    assert settings.data_folder == Path("data_folder").resolve()
    assert settings.snapshot_folder == Path("data_folder/snapshot").resolve()
    assert settings.user_file_path == Path("data_folder/users.json").resolve()
