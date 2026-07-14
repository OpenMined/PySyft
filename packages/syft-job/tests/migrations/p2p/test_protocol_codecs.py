"""Structure, coverage, and behavior of the ProtocolCodecs behind JobStorage.

A codec is selected by PROTOCOL version and may serve several of them
(``protocol_versions``); it also carries its own ``version``. The invariants
here are about protocol-version coverage: every protocol version the registry
understands is handled by exactly one codec.
"""

import importlib
import inspect
import pkgutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from syft_job import protocolcodecs
from syft_job.config import SyftJobConfig
from syft_job.job_storage import JobRef, JobStorage
from syft_job.migrations import job_registry
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION
from syft_job.models import JobStatus, JobSubmissionMetadata
from syft_job.models.job_state.v1 import JobStateV1
from syft_job.protocolcodecs import ProtocolCodec

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def _storage(tmp_path: Path) -> JobStorage:
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    return JobStorage(config=config)


def _mock_submission_metadata(name: str) -> JobSubmissionMetadata:
    return JobSubmissionMetadata(
        name=name,
        type="bash",
        submitted_by=DS_EMAIL,
        datasite_email=DO_EMAIL,
        submitted_at=datetime.now(timezone.utc),
        files=["script.sh"],
    )


# -- structural invariant ------------------------------------------------------
def test_codec_versions_unique_and_protocol_versions_disjoint(tmp_path: Path):
    codecs = _storage(tmp_path).codecs

    codec_versions = [c.version for c in codecs]
    assert len(codec_versions) == len(set(codec_versions)), (
        "codec versions must be unique"
    )

    # Each protocol version is handled by at most one codec.
    seen: set[str] = set()
    for codec in codecs:
        for protocol_version in codec.protocol_versions:
            assert protocol_version not in seen, (
                f"protocol {protocol_version} handled by more than one codec"
            )
            seen.add(protocol_version)

    # The current protocol is handled by the last (newest) codec.
    assert JOB_PROTOCOL_VERSION in codecs[-1].protocol_versions


def _all_concrete_codec_classes() -> set[type[ProtocolCodec]]:
    """Every concrete ProtocolCodec subclass defined in the protocolcodecs package."""
    for module_info in pkgutil.iter_modules(protocolcodecs.__path__):
        importlib.import_module(f"{protocolcodecs.__name__}.{module_info.name}")

    def descendants(cls: type) -> set[type]:
        subs = set(cls.__subclasses__())
        return subs.union(*(descendants(s) for s in subs))

    return {cls for cls in descendants(ProtocolCodec) if not inspect.isabstract(cls)}


# -- registration --------------------------------------------------------------
def test_all_codecs_in_codebase_are_registered_in_job_storage(tmp_path: Path):
    registered_codecs = {type(codec) for codec in _storage(tmp_path).codecs}
    defined_codecs = _all_concrete_codec_classes()

    missing = defined_codecs - registered_codecs
    assert not missing, (
        f"codec(s) defined but not registered in JobStorage/CODECS: "
        f"{sorted(cls.__name__ for cls in missing)}"
    )


# -- coverage ------------------------------------------------------------------
def test_every_known_protocol_version_is_covered_by_exactly_one_codec(tmp_path: Path):
    storage = _storage(tmp_path)

    known = set(job_registry.protocol_version_history) | {JOB_PROTOCOL_VERSION}
    covered = {pv for codec in storage.codecs for pv in codec.protocol_versions}
    assert covered == known

    # Every known protocol version resolves to a codec.
    for protocol_version in known:
        assert storage._codec_for(protocol_version) is not None


# -- behavior 1: on-disk format per codec --------------------------------------
def test_v0_writes_flat_no_identity_v1_nests_with_identity(tmp_path: Path):
    storage = _storage(tmp_path)
    state = JobStateV1(status=JobStatus.APPROVED)

    p0 = storage.write_state(JobRef(DO_EMAIL, DS_EMAIL, "flat.job", "0"), state)
    p1 = storage.write_state(JobRef(DO_EMAIL, DS_EMAIL, "nested.job", "1"), state)

    # protocol 0: review/<ds>/<job>/state.yaml (no v<n> segment), identity stripped
    assert p0.parent.parent.name == DS_EMAIL
    raw0 = yaml.safe_load(p0.read_text())
    assert "canonical_name" not in raw0 and "version" not in raw0

    # protocol 1: review/<ds>/v1/<job>/state.yaml, identity fields present
    assert p1.parent.parent.name == "v1"
    raw1 = yaml.safe_load(p1.read_text())
    assert raw1["canonical_name"] == "JobState" and raw1["version"] == "1"

    # both round-trip back to the latest version
    for ref in (
        JobRef(DO_EMAIL, DS_EMAIL, "flat.job", "0"),
        JobRef(DO_EMAIL, DS_EMAIL, "nested.job", "1"),
    ):
        loaded = storage.read_state(ref)
        assert loaded.status == JobStatus.APPROVED
        assert loaded.version == job_registry.latest_version("JobState")


# -- behavior 2: each codec scans only its own layout --------------------------
def test_scan_partitions_by_layout(tmp_path: Path):
    storage = _storage(tmp_path)
    v0_codec, v1_codec = storage._codec_for("0"), storage._codec_for("1")

    storage.write_submission(
        JobRef(DO_EMAIL, DS_EMAIL, "flat.job", "0"),
        _mock_submission_metadata("flat.job"),
    )
    storage.write_submission(
        JobRef(DO_EMAIL, DS_EMAIL, "nested.job", "1"),
        _mock_submission_metadata("nested.job"),
    )

    v0_refs = list(v0_codec.iter_submission_refs(DO_EMAIL))
    v1_refs = list(v1_codec.iter_submission_refs(DO_EMAIL))
    assert [(r.job_name, r.protocol_version) for r in v0_refs] == [("flat.job", "0")]
    assert [(r.job_name, r.protocol_version) for r in v1_refs] == [("nested.job", "1")]

    # JobStorage unions the codecs with no duplicates.
    all_refs = list(storage.iter_submission_refs(DO_EMAIL))
    assert {(r.job_name, r.protocol_version) for r in all_refs} == {
        ("flat.job", "0"),
        ("nested.job", "1"),
    }
    assert len(all_refs) == 2
