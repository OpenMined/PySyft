import re
from typing import Optional

from pydantic import BaseModel, Field
from pathlib import Path

SYFT_DATASETS_FOLDER_NAME = "syft_datasets"
METADATA_FILENAME = "dataset.yaml"
PRIVATE_METADATA_FILENAME = "private_metadata.yaml"

# Datasets under public/syft_datasets/ and private/syft_datasets/ live inside a
# protocol-version segment ("v1", "v2", ...); protocol 0 (<= 0.1.20) had none.
# The naming convention is shared; the per-protocol layout that uses it lives in
# the codecs' DatasetConfig.
PROTOCOL_DIR_RE = re.compile(r"^v\d+$")


def is_protocol_dir_name(name: str) -> bool:
    return PROTOCOL_DIR_RE.match(name) is not None


def protocol_dir_name(protocol_version: str) -> Optional[str]:
    """The path segment for a protocol version; None for protocol 0."""
    return None if protocol_version == "0" else f"v{protocol_version}"


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
