"""Structure, coverage, and behavior of the ProtocolCodecs behind DatasetStorage.

A codec is selected by PROTOCOL version and may serve several of them
(``protocol_versions``); it also carries its own ``version``. The invariants
here are about protocol-version coverage: every protocol version the registry
understands is handled by exactly one codec.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

import yaml
from syft_datasets import protocolcodecs
from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_storage import DatasetRef, DatasetStorage
from syft_datasets.migrations import dataset_registry
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_datasets.models import Dataset
from syft_datasets.protocolcodecs import ProtocolCodec

DO_EMAIL = "do@test.org"


def _storage(tmp_path: Path) -> DatasetStorage:
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    config = SyftBoxConfig(syftbox_folder=syftbox, email=DO_EMAIL)
    return DatasetStorage(config=config)


def _mock_dataset(storage: DatasetStorage, ref: DatasetRef) -> Dataset:
    cfg = storage.config
    return Dataset(
        name=ref.name,
        mock_url=cfg.get_mock_url_for_my_dataset(ref.name, ref.protocol_version),
        private_url=cfg.get_private_url_for_my_dataset(ref.name, ref.protocol_version),
    )


# -- structural invariant ------------------------------------------------------
def test_codec_versions_unique_and_protocol_versions_disjoint(tmp_path: Path):
    codecs = _storage(tmp_path).codecs

    codec_versions = [c.version for c in codecs]
    assert len(codec_versions) == len(set(codec_versions)), (
        "codec versions must be unique"
    )

    seen: set[str] = set()
    for codec in codecs:
        for protocol_version in codec.protocol_versions:
            assert protocol_version not in seen, (
                f"protocol {protocol_version} handled by more than one codec"
            )
            seen.add(protocol_version)

    # The current protocol is handled by the last (newest) codec.
    assert DATASET_PROTOCOL_VERSION in codecs[-1].protocol_versions


def _all_concrete_codec_classes() -> set[type[ProtocolCodec]]:
    for module_info in pkgutil.iter_modules(protocolcodecs.__path__):
        importlib.import_module(f"{protocolcodecs.__name__}.{module_info.name}")

    def descendants(cls: type) -> set[type]:
        subs = set(cls.__subclasses__())
        return subs.union(*(descendants(s) for s in subs))

    return {cls for cls in descendants(ProtocolCodec) if not inspect.isabstract(cls)}


# -- registration --------------------------------------------------------------
def test_all_codecs_in_codebase_are_registered_in_dataset_storage(tmp_path: Path):
    registered_codecs = {type(codec) for codec in _storage(tmp_path).codecs}
    defined_codecs = _all_concrete_codec_classes()

    missing = defined_codecs - registered_codecs
    assert not missing, (
        f"codec(s) defined but not registered in DatasetStorage/CODECS: "
        f"{sorted(cls.__name__ for cls in missing)}"
    )


# -- coverage ------------------------------------------------------------------
def test_every_known_protocol_version_is_covered_by_exactly_one_codec(tmp_path: Path):
    storage = _storage(tmp_path)

    known = set(dataset_registry.protocol_version_history) | {DATASET_PROTOCOL_VERSION}
    covered = {pv for codec in storage.codecs for pv in codec.protocol_versions}
    assert covered == known

    for protocol_version in known:
        assert storage._codec_for(protocol_version) is not None


# -- behavior 1: on-disk format per codec --------------------------------------
def test_v0_writes_flat_no_identity_v1_nests_with_identity(tmp_path: Path):
    storage = _storage(tmp_path)

    ref0 = DatasetRef(DO_EMAIL, "flat", "0")
    ref1 = DatasetRef(DO_EMAIL, "nested", "1")
    p0 = storage.write_dataset(ref0, _mock_dataset(storage, ref0))
    p1 = storage.write_dataset(ref1, _mock_dataset(storage, ref1))

    # protocol 0: public/syft_datasets/<name>/dataset.yaml (no v<n>), identity stripped
    assert p0.parent.parent.name == "syft_datasets"
    raw0 = yaml.safe_load(p0.read_text())
    assert "canonical_name" not in raw0 and "version" not in raw0

    # protocol 1: public/syft_datasets/v1/<name>/dataset.yaml, identity present
    assert p1.parent.parent.name == "v1"
    raw1 = yaml.safe_load(p1.read_text())
    assert raw1["canonical_name"] == "Dataset" and raw1["version"] == "1"

    for ref in (ref0, ref1):
        loaded = storage.read_dataset(ref)
        assert loaded.name == ref.name
        assert loaded.version == dataset_registry.latest_version("Dataset")


# -- behavior 2: each codec scans only its own layout --------------------------
def test_scan_partitions_by_layout(tmp_path: Path):
    storage = _storage(tmp_path)
    v0_codec, v1_codec = storage._codec_for("0"), storage._codec_for("1")

    ref0 = DatasetRef(DO_EMAIL, "flat", "0")
    ref1 = DatasetRef(DO_EMAIL, "nested", "1")
    storage.write_dataset(ref0, _mock_dataset(storage, ref0))
    storage.write_dataset(ref1, _mock_dataset(storage, ref1))

    v0_refs = list(v0_codec.iter_dataset_refs(DO_EMAIL))
    v1_refs = list(v1_codec.iter_dataset_refs(DO_EMAIL))
    assert [(r.name, r.protocol_version) for r in v0_refs] == [("flat", "0")]
    assert [(r.name, r.protocol_version) for r in v1_refs] == [("nested", "1")]

    all_refs = list(storage.iter_dataset_refs(DO_EMAIL))
    assert {(r.name, r.protocol_version) for r in all_refs} == {
        ("flat", "0"),
        ("nested", "1"),
    }
    assert len(all_refs) == 2
