from pathlib import Path

from syft_perms import SyftPermContext


def set_mock_dataset_permissions(
    syftbox_folder: Path,
    owner_email: str,
    mock_dir: Path,
    users: list[str],
) -> None:
    """Create syft.pub.yaml in mock dir granting read access to users."""
    datasite = syftbox_folder / owner_email
    ctx = SyftPermContext(datasite=datasite)
    rel_path = str(mock_dir.relative_to(datasite)) + "/"
    folder = ctx.open(rel_path)
    for user in users:
        folder.grant_read_access(user)


def set_private_dataset_permissions(
    syftbox_folder: Path,
    owner_email: str,
    private_dir: Path,
) -> None:
    """Create syft.pub.yaml in private dir granting read only to owner.

    Skips if private_dir is not under the datasite (e.g., stored in a
    separate 'private/' folder outside the datasite tree).
    """
    datasite = syftbox_folder / owner_email
    try:
        rel_path = str(private_dir.relative_to(datasite)) + "/"
    except ValueError:
        # Private dir is outside the datasite - no permissions needed
        return
    ctx = SyftPermContext(datasite=datasite)
    folder = ctx.open(rel_path)
    folder.grant_read_access(owner_email)
