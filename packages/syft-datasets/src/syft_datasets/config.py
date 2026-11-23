import re

from pydantic import BaseModel, Field
from pathlib import Path

SYFT_DATASETS_FOLDER_NAME = "syft_datasets"
METADATA_FILENAME = "dataset.yaml"
PRIVATE_METADATA_FILENAME = "private_metadata.yaml"

# Datasets under public/syft_datasets/ and private/syft_datasets/ live inside a
# protocol-version segment ("v0", "v1", ...).
# The naming convention is shared; the per-protocol layout that uses it lives in
# the codecs' DatasetConfig.
PROTOCOL_DIR_RE = re.compile(r"^v\d+$")


def is_protocol_dir_name(name: str) -> bool:
    return PROTOCOL_DIR_RE.match(name) is not None


def protocol_dir_name(protocol_version: str) -> str:
    """The path segment for a protocol version."""
    return f"v{protocol_version}"


class SyftBoxConfig(BaseModel):
    """Environment: which SyftBox and whose datasite. Protocol-agnostic.

    Dataset on-disk layout (folders, v<n> segments, filenames) is *not* here; it
    is versioned per protocol and owned by the codecs' DatasetConfig. This only
    knows the SyftBox tree, which is stable across dataset protocol versions.
    """

    syftbox_folder: Path = Field(
        ..., description="Path to the SyftBox folder on the local filesystem."
    )
    email: str = Field(..., description="Email associated with the SyftBox.")

    @property
    def private_dir(self) -> Path:
        return self.syftbox_folder / self.email / "private"

    @property
    def public_dir(self) -> Path:
        return self.syftbox_folder / self.email / "public"

    def datasite_public_root(self, datasite: str) -> Path:
        """The public/ root under a datasite (no dataset layout applied)."""
        return self.syftbox_folder / datasite / "public"

    def datasite_private_root(self, datasite: str) -> Path:
        """The private/ root under a datasite (no dataset layout applied)."""
        return self.syftbox_folder / datasite / "private"
