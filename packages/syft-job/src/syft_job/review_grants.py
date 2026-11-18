"""The read grant of a data scientist on its review directory.

A job runs as the data owner, so it can write any file into its review
directory. The data scientist therefore reads only named files there. Every
other file reaches the data scientist through a per-file grant at release.
"""

from pathlib import Path

from syft_permissions import RuleSet
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME
from syft_perms import SyftPermContext

# The file the DS reads before anything is shared, in each job folder layout:
# <job>/ for protocol 0, v<n>/<job>/ after. A test checks every protocol is here.
DS_READABLE_REVIEW_PATTERNS = (
    "*/state.yaml",
    "v1/*/state.yaml",
    "v2/*/state.yaml",
    "v3/*/state.yaml",
)

# The read grant on the whole folder that earlier versions wrote.
_FOLDER_PATTERN = "**"


def grant_ds_state_read(
    ctx: SyftPermContext, ds_review_dir: Path, ds_email: str
) -> None:
    """Give ``ds_email`` read on the state of each job."""
    folder = str(ds_review_dir.relative_to(ctx.datasite))
    for pattern in DS_READABLE_REVIEW_PATTERNS:
        ctx.modifier.add_permission_for_pattern(folder, pattern, "read", ds_email)


def replace_ds_folder_read(datasite: Path, ds_review_dir: Path, ds_email: str) -> bool:
    """Replace an earlier read grant of ``ds_email`` on the whole folder.

    Returns True if the folder held that grant. A folder without it does not
    change, so the call adds no grant that the data owner did not give.
    """
    if not _drop_folder_read(ds_review_dir, ds_email):
        return False
    ctx = SyftPermContext(datasite=datasite)
    grant_ds_state_read(ctx, ds_review_dir, ds_email)
    return True


def _drop_folder_read(ds_review_dir: Path, ds_email: str) -> bool:
    """Remove ``ds_email`` from the folder rule of the folder's own permission file.

    Returns True if it was there. A folder rule left with no user is removed.
    Not syft-perms' revoke: with no file here it edits a parent's, wildcards too.
    """
    path = ds_review_dir / PERMISSION_FILE_NAME
    if not path.exists():
        return False
    ruleset = RuleSet.load(path)
    dropped = False
    for rule in list(ruleset.rules):
        if rule.pattern != _FOLDER_PATTERN or ds_email not in rule.access.read:
            continue
        rule.access.read = [u for u in rule.access.read if u != ds_email]
        dropped = True
        access = rule.access
        if not (access.admin or access.write or access.read):
            ruleset.rules.remove(rule)
    if dropped:
        ruleset.save(path)
    return dropped
