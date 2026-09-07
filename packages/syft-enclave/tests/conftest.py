from pathlib import Path
from typing import Optional

import pytest

PRIVATE_DATASETS_REL = Path("private") / "syft_datasets"


def _private_dataset_dirs(
    syftbox_folder: Path, owner_email: str, tag: str
) -> list[Path]:
    """Every layout of one private dataset: the flat one and each v<n> one."""
    base = syftbox_folder / owner_email / PRIVATE_DATASETS_REL
    if not base.is_dir():
        return []
    candidates = [base / tag]
    candidates += [d / tag for d in sorted(base.glob("v*")) if d.is_dir()]
    return [p for p in candidates if p.is_dir()]


@pytest.fixture
def private_dataset_dir():
    """Find the private directory of a dataset, whatever protocol layout holds it.

    The layout of a private dataset is `private/syft_datasets/[v<n>/]<tag>`, and
    the segment depends on the protocol version of the copy. A test asserts that
    the data arrived, so it must not name one version.

    Returns a callable `(syftbox_folder, owner_email, tag) -> Path | None`. The
    callable raises if more than one layout holds the dataset.
    """

    def _find(syftbox_folder: Path, owner_email: str, tag: str) -> Optional[Path]:
        dirs = _private_dataset_dirs(syftbox_folder, owner_email, tag)
        assert len(dirs) <= 1, f"More than one layout holds {tag!r}: {dirs}"
        return dirs[0] if dirs else None

    return _find
