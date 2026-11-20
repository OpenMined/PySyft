"""The read grant of a data scientist on its review directory.

A job runs as the data owner, so it can write any file into its review
directory. The data scientist therefore reads only named files there. Every
other file reaches the data scientist through a per-file grant at release.
"""

from pathlib import Path

from syft_perms import SyftPermContext

# The file the DS reads before anything is shared, in each job folder layout
# v<n>/<job>/. A test checks every protocol is here.
DS_READABLE_REVIEW_PATTERNS = ("v0/*/state.yaml",)


def grant_ds_state_read(
    ctx: SyftPermContext, ds_review_dir: Path, ds_email: str
) -> None:
    """Give ``ds_email`` read on the state of each job."""
    folder = str(ds_review_dir.relative_to(ctx.datasite))
    for pattern in DS_READABLE_REVIEW_PATTERNS:
        ctx.modifier.add_permission_for_pattern(folder, pattern, "read", ds_email)
