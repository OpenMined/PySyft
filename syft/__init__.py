"""
syft - A unified client for secure file syncing (formerly published as ``syft-client``).

"""

import logging
from pathlib import Path

# Default logging for the syft namespace. Only installs a handler if
# nothing is already configured -- callers who set up their own logging keep
# full control. Records still propagate to the root logger so test fixtures
# (pytest's caplog) and user-configured root handlers can observe them; in
# default Python the root logger has no handler so there's no double print.
_logger = logging.getLogger("syft")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

from syft.version import SYFT_VERSION as __version__  # noqa: F401, E402
from syft.sync.login import login_do, login_ds, login  # noqa: F401, E402
from syft.utils import (  # noqa: F401, E402
    resolve_path,
    resolve_dataset_file_path,
    resolve_dataset_files_path,
    load_dataset_code,
    bug_report,
)
from syft.gdrive_utils import (  # noqa: F401, E402
    download_from_gdrive,
    credentials_to_token,
    delete_remote_syftbox,
)
from syft.sync.utils.syftbox_utils import (  # noqa: F401, E402
    delete_syftbox,
    delete_local_syftbox,
)
from syft.migrations.history import register_historic_schemas  # noqa: E402

# Import the versioned model modules explicitly so registration is intentional,
# not a side-effect of whatever login happened to pull in first.
import syft.sync.version.version_info  # noqa: F401, E402
import syft.sync.messages.proposed_filechange  # noqa: F401, E402
import syft.sync.events.file_change_event  # noqa: F401, E402

register_historic_schemas()

SYFT_DIR = Path(__file__).parent.parent
CREDENTIALS_DIR = SYFT_DIR / "credentials"
