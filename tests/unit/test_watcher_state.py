import json

import pytest

from syft.sync.checkpoints.watcher_state import (
    WATCHER_STATE_VERSION,
    WatcherCheckpoint,
)
from syft.sync.utils.syftbox_utils import compress_data


def test_watcher_state_of_later_version_refused():
    later_version = WATCHER_STATE_VERSION + 1
    state = {"version": later_version, "events": [], "new_setting": True}
    data = compress_data(json.dumps(state).encode("utf-8"))

    with pytest.raises(ValueError, match=f"version {later_version}"):
        WatcherCheckpoint.from_compressed_data(data)
