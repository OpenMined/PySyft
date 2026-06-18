"""Path helpers for the inference service.

The inference server runs in a separate process from the enclave runner and
must never share a SyftboxManager with it — it only reads/writes well-known
paths inside the syftbox folder, which these helpers compute.
"""

from pathlib import Path

from syft_client.sync.syftbox_manager import get_jupyter_default_syftbox_folder
from syft_datasets.config import SyftBoxConfig


def default_syftbox_folder(email: str) -> Path:
    """The folder SyftboxManagerConfig.for_jupyter (used by for_enclave) picks."""
    return get_jupyter_default_syftbox_folder(email)


def private_dataset_dir(
    syftbox_folder: Path | str, datasite: str, dataset_name: str
) -> Path:
    """Private dir of *dataset_name* on *datasite* inside *syftbox_folder*."""
    config = SyftBoxConfig(syftbox_folder=Path(syftbox_folder), email=datasite)
    return config.private_dir_for_my_dataset(dataset_name)


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
