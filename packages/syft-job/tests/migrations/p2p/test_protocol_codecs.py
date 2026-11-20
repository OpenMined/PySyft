"""Structure and coverage of the ProtocolCodecs behind JobStorage.

A codec is selected by PROTOCOL version and may serve several of them
(``protocol_versions``); it also carries its own ``version``. The invariants
here are about protocol-version coverage: every protocol version the registry
understands is handled by exactly one codec.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

from syft_job import protocolcodecs
from syft_job.config import SyftJobConfig
from syft_job.job_storage import JobStorage
from syft_job.migrations import job_registry
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION
from syft_job.protocolcodecs import ProtocolCodec

DO_EMAIL = "do@test.org"


def _storage(tmp_path: Path) -> JobStorage:
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    config = SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    return JobStorage(config=config)


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
