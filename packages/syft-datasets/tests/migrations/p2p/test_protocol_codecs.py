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
from syft_datasets.config import SyftBoxConfig, protocol_dir_name
from syft_datasets.dataset_storage import DatasetRef, DatasetStorage
from syft_datasets.migrations import dataset_registry
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_datasets.models import Dataset, PrivateDatasetConfig
from syft_datasets.protocolcodecs import ProtocolCodec
from syft_datasets.url import SyftBoxURL

DO_EMAIL = "do@test.org"


def _storage(tmp_path: Path) -> DatasetStorage:
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    config = SyftBoxConfig(syftbox_folder=syftbox, email=DO_EMAIL)
    return DatasetStorage(config=config)


def _mock_dataset(storage: DatasetStorage, ref: DatasetRef) -> Dataset:
    folder = storage.config.syftbox_folder
    return Dataset(
        name=ref.name,
        mock_url=SyftBoxURL.from_path(storage.public_dataset_dir(ref), folder),
        private_url=SyftBoxURL.from_path(storage.private_dataset_dir(ref), folder),
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


# -- behavior: on-disk format of the current protocol ---------------------------
def test_the_current_protocol_nests_under_its_segment_with_identity(tmp_path: Path):
    storage = _storage(tmp_path)
    ref = DatasetRef(DO_EMAIL, "nested", DATASET_PROTOCOL_VERSION)
    path = storage.write_dataset_metadata(ref, _mock_dataset(storage, ref))

    # public/syft_datasets/v<n>/<name>/dataset.yaml, identity present
    assert path.parent.parent.name == protocol_dir_name(DATASET_PROTOCOL_VERSION)
    raw = yaml.safe_load(path.read_text())
    assert raw["canonical_name"] == "Dataset"
    assert raw["version"] == dataset_registry.latest_version("Dataset")

    loaded = storage.read_dataset(ref)
    assert loaded.name == ref.name
    assert loaded.version == dataset_registry.latest_version("Dataset")


def test_wire_bytes_match_write(tmp_path: Path):
    storage = _storage(tmp_path)
    ref = DatasetRef(DO_EMAIL, "nested", DATASET_PROTOCOL_VERSION)
    dataset = _mock_dataset(storage, ref)
    config = PrivateDatasetConfig(uid=dataset.uid, data_dir=Path("/secret/local/path"))

    path = storage.write_private_config(ref, config)

    assert storage.wire_bytes(ref, config) == path.read_bytes()
    raw = yaml.safe_load(path.read_text())
    assert raw["canonical_name"] == "PrivateDatasetConfig"
    assert raw["version"] == dataset_registry.latest_version("PrivateDatasetConfig")
