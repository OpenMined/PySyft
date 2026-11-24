"""A checkpoint or rolling state from a later client is refused, not restored.

Both models carry a `version` field that nothing read. A later client can change
what a field means while the object still parses, because pydantic accepts a
document that holds every field it knows. The restore would then be wrong and
silent.

Refusing is cheap here. Every load site already falls back to a download of all
events when a checkpoint fails to load, so an unusable checkpoint costs one slow
cold start and nothing else.
"""

import json

import pytest
from syft.sync.checkpoints.checkpoint import (
    CHECKPOINT_VERSION,
    INCREMENTAL_CHECKPOINT_VERSION,
    Checkpoint,
    IncrementalCheckpoint,
)
from syft.sync.checkpoints.rolling_state import (
    ROLLING_STATE_VERSION,
    RollingState,
)
from syft.sync.utils.syftbox_utils import compress_data

EMAIL = "alice@example.com"


def _checkpoint(**kwargs) -> Checkpoint:
    return Checkpoint(email=EMAIL, **kwargs)


def _incremental(**kwargs) -> IncrementalCheckpoint:
    return IncrementalCheckpoint(email=EMAIL, sequence_number=1, **kwargs)


def _rolling(**kwargs) -> RollingState:
    return RollingState(email=EMAIL, base_checkpoint_timestamp=1.0, **kwargs)


def test_a_checkpoint_round_trips():
    loaded = Checkpoint.from_compressed_data(_checkpoint().as_compressed_data())
    assert loaded.version == CHECKPOINT_VERSION


def test_an_incremental_checkpoint_round_trips():
    loaded = IncrementalCheckpoint.from_compressed_data(
        _incremental().as_compressed_data()
    )
    assert loaded.version == INCREMENTAL_CHECKPOINT_VERSION


def test_a_rolling_state_round_trips():
    loaded = RollingState.from_compressed_data(_rolling().as_compressed_data())
    assert loaded.version == ROLLING_STATE_VERSION


def test_a_later_checkpoint_is_refused():
    data = _checkpoint(version=CHECKPOINT_VERSION + 1).as_compressed_data()
    with pytest.raises(ValueError, match=str(CHECKPOINT_VERSION + 1)):
        Checkpoint.from_compressed_data(data)


def test_a_later_incremental_checkpoint_is_refused():
    later_version = INCREMENTAL_CHECKPOINT_VERSION + 1
    data = _incremental(version=later_version).as_compressed_data()
    with pytest.raises(ValueError, match=str(later_version)):
        IncrementalCheckpoint.from_compressed_data(data)


def test_a_later_rolling_state_is_refused():
    data = _rolling(version=ROLLING_STATE_VERSION + 1).as_compressed_data()
    with pytest.raises(ValueError, match=str(ROLLING_STATE_VERSION + 1)):
        RollingState.from_compressed_data(data)


def test_a_version_1_checkpoint_loads_its_text_content():
    # Version 1 stored content as text. "true" is valid base64 too, so a base64
    # read would give other bytes without an error.
    version_1 = {
        "version": 1,
        "timestamp": 1.0,
        "email": EMAIL,
        "last_event_timestamp": 1.0,
        "files": [{"path": "public/a.txt", "hash": "abc", "content": "true"}],
    }
    data = compress_data(json.dumps(version_1).encode("utf-8"))

    loaded = Checkpoint.from_compressed_data(data)

    assert loaded.version == CHECKPOINT_VERSION
    assert loaded.files[0].content == b"true"
    reloaded = Checkpoint.from_compressed_data(loaded.as_compressed_data())
    assert reloaded.files[0].content == b"true"
