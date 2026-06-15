"""
syft_client - A unified client for secure file syncing
"""

import logging
from pathlib import Path

# Default logging for the syft_client namespace. Only installs a handler if
# nothing is already configured -- callers who set up their own logging keep
# full control. Records still propagate to the root logger so test fixtures
# (pytest's caplog) and user-configured root handlers can observe them; in
# default Python the root logger has no handler so there's no double print.
_logger = logging.getLogger("syft_client")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

from syft_client.version import SYFT_CLIENT_VERSION as __version__  # noqa: F401, E402
from syft_client.sync.login import login_do, login_ds, login  # noqa: F401, E402
from syft_client.utils import (  # noqa: F401, E402
    resolve_path,
    resolve_dataset_file_path,
    resolve_dataset_files_path,
    load_dataset_code,
    bug_report,
)
from syft_client.gdrive_utils import (  # noqa: F401, E402
    download_from_gdrive,
    credentials_to_token,
    delete_remote_syftbox,
)
from syft_client.sync.utils.syftbox_utils import (  # noqa: F401, E402
    delete_syftbox,
    delete_local_syftbox,
)

SYFT_CLIENT_DIR = Path(__file__).parent.parent
CREDENTIALS_DIR = SYFT_CLIENT_DIR / "credentials"
