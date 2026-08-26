"""Path helpers for the inference service.

The inference server runs in a separate process from the enclave runner and
must never share a SyftboxManager with it — it only reads/writes well-known
paths inside the syftbox folder, which these helpers compute.
"""

from pathlib import Path

from syft.sync.syftbox_manager import get_jupyter_default_syftbox_folder
from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_ref import DatasetNotFoundError, DatasetRef
from syft_datasets.dataset_storage import DatasetStorage


def default_syftbox_folder(email: str) -> Path:
    """The folder SyftboxManagerConfig.for_jupyter (used by for_enclave) picks."""
    return get_jupyter_default_syftbox_folder(email)


def resolve_private_dataset_dir(storage: DatasetStorage, owner: str, name: str) -> Path:
    """Private dir at the dataset's actual on-disk protocol layout.

    A dataset may live at protocol 0 (flat) or under a ``v<n>`` segment; the
    written layout depends on what the audience can read, not the current
    default. Datasets not yet on disk (e.g. weights still syncing) fall back to
    the widest-compatible protocol — where a peer running any current release
    writes them for us.
    """
    try:
        ref = storage.find_dataset_ref(owner, name)
    except DatasetNotFoundError:
        (widest,) = storage.target_protocol_versions_for_peers(None)
        ref = DatasetRef(owner=owner, name=name, protocol_version=widest)
    return storage.private_dataset_dir(ref)


def private_dataset_dir(
    syftbox_folder: Path | str, datasite: str, dataset_name: str
) -> Path:
    """Private dir of *dataset_name* on *datasite* inside *syftbox_folder*."""
    config = SyftBoxConfig(syftbox_folder=Path(syftbox_folder), email=datasite)
    storage = DatasetStorage(config=config)
    return resolve_private_dataset_dir(storage, datasite, dataset_name)


def find_checkpoint_dir(weights_dir: Path | str) -> Path | None:
    """The single subdirectory of the weights dataset holding the checkpoint."""
    weights_dir = Path(weights_dir)
    if not weights_dir.is_dir():
        return None
    subdirs = [p for p in weights_dir.iterdir() if p.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else None


def weights_ready(weights_dir: Path | str) -> bool:
    """True once the synced weights dataset has a tokenizer and a checkpoint."""
    weights_dir = Path(weights_dir)
    has_tokenizer = (weights_dir / "tokenizer.model").is_file()
    return has_tokenizer and find_checkpoint_dir(weights_dir) is not None
