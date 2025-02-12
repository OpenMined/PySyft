from syftbox.lib.exceptions import SyftBoxException


class FatalSyncError(SyftBoxException):
    """Base exception to signal sync should be interrupted."""

    pass


class SyncEnvironmentError(FatalSyncError):
    """the sync environment is corrupted (e.g. sync folder deleted), syncing cannot continue."""

    pass


class SyncValidationError(SyftBoxException):
    pass
