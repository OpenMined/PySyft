import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from syftbox.lib.hash import hash_file
from syftbox.server.db.file_store import FileStore
from syftbox.server.settings import ServerSettings


def test_put_atomic(tmpdir):
    settings = ServerSettings.from_data_folder(tmpdir)
    syft_path = Path("test.txt")
    system_path = settings.snapshot_folder / syft_path
    user = "example@example.com"

    with ThreadPoolExecutor(max_workers=5) as executor:
        # TODO: add permissions
        executor.map(
            lambda _: FileStore(settings).put(
                syft_path, uuid.uuid4().bytes, user, check_permission=None, skip_permission_check=True
            ),
            range(25),
        )

    assert system_path.exists()
    metadata = FileStore(settings).get_metadata(syft_path, user, skip_permission_check=True)
    assert metadata.hash_bytes == hash_file(system_path).hash_bytes
