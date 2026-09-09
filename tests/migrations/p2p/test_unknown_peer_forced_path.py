"""A forced submission reports the protocol version that it assumes.

A peer of unknown version is refused before this point. A caller that passes
``raise_on_unknown=False`` skips that refusal. The storage then assumes the
current protocol.

If the peer speaks an earlier protocol, it does not scan this layout. The job or
the dataset never arrives, so the storage writes a warning.
"""

import logging
from pathlib import Path

from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_storage import DatasetStorage
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_job import SyftJobConfig
from syft_job.job_storage import JobStorage
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def _job_storage(tmp_path: Path) -> JobStorage:
    config = SyftJobConfig(
        syftbox_folder=tmp_path / "SyftBox", current_user_email=DS_EMAIL
    )
    (tmp_path / "SyftBox" / DS_EMAIL).mkdir(parents=True, exist_ok=True)
    return JobStorage(config=config, peer_schemas={})


def _dataset_storage(tmp_path: Path) -> DatasetStorage:
    config = SyftBoxConfig(syftbox_folder=tmp_path / "SyftBox", email=DO_EMAIL)
    (tmp_path / "SyftBox" / DO_EMAIL).mkdir(parents=True, exist_ok=True)
    return DatasetStorage(config=config, peer_schemas={})


def test_a_forced_job_reports_the_assumed_protocol(tmp_path, caplog):
    storage = _job_storage(tmp_path)
    with caplog.at_level(logging.WARNING):
        version = storage.negotiated_protocol_version_for_peer(
            DO_EMAIL, raise_on_unknown=False
        )
    assert version == JOB_PROTOCOL_VERSION
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert DO_EMAIL in messages
    assert "earlier protocol" in messages


def test_a_forced_dataset_reports_the_assumed_protocol(tmp_path, caplog):
    storage = _dataset_storage(tmp_path)
    with caplog.at_level(logging.WARNING):
        version = storage.negotiated_protocol_version_for_peer(
            DS_EMAIL, raise_on_unknown=False
        )
    assert version == DATASET_PROTOCOL_VERSION
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert DS_EMAIL in messages
    assert "earlier protocol" in messages


def test_a_known_peer_reports_nothing(tmp_path, caplog):
    # The report belongs to the forced path only. A known peer is negotiated.
    from syft_migration import ProtocolSchema

    storage = _job_storage(tmp_path)
    storage.peer_schemas[DO_EMAIL] = ProtocolSchema(
        protocol_name="syft-job",
        version=JOB_PROTOCOL_VERSION,
        supported_versions={"JobState": ["1"]},
    )
    with caplog.at_level(logging.WARNING):
        storage.negotiated_protocol_version_for_peer(DO_EMAIL)
    assert caplog.records == []
