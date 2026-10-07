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


def candidate_private_dataset_dirs(
    storage: DatasetStorage, owner: str, name: str
) -> list[Path]:
    """The private dirs a dataset could occupy, newest layout first.

    One entry, the layout on disk, once the dataset is there. While it is
    absent, one entry for each protocol layout this client reads: the owner
    writes the layout its own release and audience decided, and a dataset that
    has not arrived cannot tell us which that is. A reader that waits must
    therefore watch them all.
    """
    try:
        ref = storage.find_dataset_ref(owner, name)
    except DatasetNotFoundError:
        return [
            storage.private_dataset_dir(
                DatasetRef(owner=owner, name=name, protocol_version=protocol_version)
            )
            for protocol_version in storage.supported_protocol_versions
        ]
    return [storage.private_dataset_dir(ref)]


def resolve_private_dataset_dir(storage: DatasetStorage, owner: str, name: str) -> Path:
    """Private dir at the dataset's on-disk protocol layout.

    The current layout while the dataset is absent, which is the layout this
    client writes for a dataset of its own. A reader waiting for a dataset that
    another datasite writes must use ``candidate_private_dataset_dirs``, because
    that owner may write an older layout.
    """
    return candidate_private_dataset_dirs(storage, owner, name)[0]


def resolve_weights_dir(
    syftbox_folder: Path | str, datasite: str, dataset_name: str
) -> Path:
    """The layout that holds the synced weights, or the newest candidate so far.

    Re-resolved on each poll, not fixed at startup: the weights arrive in the
    layout their owner writes, and an owner on an earlier release writes the
    flat one.
    """
    config = SyftBoxConfig(syftbox_folder=Path(syftbox_folder), email=datasite)
    storage = DatasetStorage(config=config)
    candidates = candidate_private_dataset_dirs(storage, datasite, dataset_name)
    for path in candidates:
        if weights_ready(path):
            return path
    return candidates[0]


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
