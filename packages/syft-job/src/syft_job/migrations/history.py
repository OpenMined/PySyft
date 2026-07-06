from pathlib import Path

from syft_migration import ReleaseArtifact

from .registry import job_registry

# Release artifacts of past syft-job releases. 0.1.38 (protocol 0) predates the
# artifact mechanism, so its file is hardcoded as if that release had emitted it.
HISTORY_DIR = Path(__file__).parent / "history"


def register_historic_schemas() -> None:
    """Register the release artifacts of past releases into the job registry.

    Must run after the versioned models are imported: with
    ``raise_for_unknown_objects`` an artifact listing an object version this
    release cannot load fails at import time instead of at migration time.
    """
    for path in sorted(HISTORY_DIR.glob("*.json")):
        job_registry.register_historic_release_artifact(
            ReleaseArtifact.load(path), raise_for_unknown_objects=True
        )
