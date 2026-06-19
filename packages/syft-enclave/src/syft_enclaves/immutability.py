import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

PRIVATE_DATASET_PARTS = ("private", "syft_datasets")


def is_private_dataset_path(path: str) -> bool:
    """Check if *path* points to a file inside a private dataset directory.

    Expected shape: ``<email>/private/syft_datasets/<dataset_name>/<file>``
    """
    parts = Path(path).parts
    return len(parts) >= 5 and parts[1:3] == PRIVATE_DATASET_PARTS


def make_private_dataset_immutability_filter(
    syftbox_folder: Path,
) -> Callable[[str, bool], bool]:
    """Return a pre-write filter that blocks overwrites/deletes of existing
    private dataset files on disk.

    Args:
        syftbox_folder: Root SyftBox directory where files are materialized.

    Returns:
        A callable ``(path_in_syftbox, is_delete) -> bool``.
        ``True`` allows the operation; ``False`` denies it.
    """

    def _deny_existing_private_dataset_files(
        path_in_syftbox: str, is_delete: bool
    ) -> bool:
        if not is_private_dataset_path(path_in_syftbox):
            return True

        full_path = syftbox_folder / path_in_syftbox
        if full_path.exists():
            logger.warning(
                "Immutability: denied %s of %s (file already exists)",
                "delete" if is_delete else "overwrite",
                path_in_syftbox,
            )
            return False

        return True

    return _deny_existing_private_dataset_files
