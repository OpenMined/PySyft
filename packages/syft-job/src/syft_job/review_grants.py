"""The read grant of a data scientist on its review directory.

A job runs as the data owner, so it can write any file into its review
directory. The data scientist therefore reads only named files there. Every
other file reaches the data scientist through a per-file grant at release.
"""

from pathlib import Path

from syft_permissions import Access, Rule, RuleSet
from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from .config import protocol_dir_name
from .disclosures import DISCLOSURES_FILENAME
from .protocolcodecs import CODECS

# The files that the data scientist reads before any release: the job state,
# to poll, and the data owner's grant of disclosures. Each pattern names the
# file at the job level of one protocol layout, so a file with the same name
# lower in the tree does not match.
_DS_READABLE_REVIEW_FILES = ("state.yaml", DISCLOSURES_FILENAME)
DS_READABLE_REVIEW_PATTERNS = tuple(
    "/".join(filter(None, (protocol_dir_name(version), "*", name)))
    for codec in CODECS
    for version in codec.protocol_versions
    for name in _DS_READABLE_REVIEW_FILES
)

# The read grant on the whole folder that earlier versions wrote.
_FOLDER_PATTERN = "**"


def grant_ds_review_read(ds_review_dir: Path, ds_email: str) -> None:
    """Give ``ds_email`` read on the state and the grant file of each job.

    Removes an earlier read grant of ``ds_email`` on the whole folder. The
    other rules in the file, such as the per-file grants of a release, stay.
    """
    path = ds_review_dir / PERMISSION_FILE_NAME
    ruleset = RuleSet.load(path) if path.exists() else RuleSet()
    before = ruleset.model_dump()
    _drop_folder_read(ruleset, ds_email)
    _add_file_reads(ruleset, ds_email)
    if not path.exists() or ruleset.model_dump() != before:
        ds_review_dir.mkdir(parents=True, exist_ok=True)
        ruleset.save(path)


def replace_ds_folder_read(ds_review_dir: Path, ds_email: str) -> bool:
    """Replace an earlier read grant of ``ds_email`` on the whole folder.

    Returns True if the folder held that grant. A folder without it does not
    change, so the call adds no grant that the data owner did not give.
    """
    path = ds_review_dir / PERMISSION_FILE_NAME
    if not path.exists():
        return False
    ruleset = RuleSet.load(path)
    if not _drop_folder_read(ruleset, ds_email):
        return False
    _add_file_reads(ruleset, ds_email)
    ruleset.save(path)
    return True


def _drop_folder_read(ruleset: RuleSet, ds_email: str) -> bool:
    """Remove ``ds_email`` from the read list of the folder rule.

    A folder rule that this leaves with no user is removed.
    """
    dropped = False
    for rule in list(ruleset.rules):
        if rule.pattern != _FOLDER_PATTERN or ds_email not in rule.access.read:
            continue
        rule.access.read = [u for u in rule.access.read if u != ds_email]
        dropped = True
        access = rule.access
        if not (access.admin or access.write or access.read):
            ruleset.rules.remove(rule)
    return dropped


def _add_file_reads(ruleset: RuleSet, ds_email: str) -> None:
    for pattern in DS_READABLE_REVIEW_PATTERNS:
        rule = next((r for r in ruleset.rules if r.pattern == pattern), None)
        if rule is None:
            rule = Rule(pattern=pattern, access=Access())
            ruleset.rules.append(rule)
        if ds_email not in rule.access.read:
            rule.access.read.append(ds_email)
