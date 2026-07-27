"""Generate a p2p backward-compatibility fixture for the current syft-job release.

Run on EVERY release, after bumping the version:

    uv run python scripts/generate_release_fixture.py

Writes a SyftBox tree exactly as this release serializes jobs to disk, into

    tests/migrations/p2p/fixtures/syft_job-<version>-protocol<p>_syftbox/

Future releases loop over these fixtures (test_older_protocol_compatibility.py)
to prove they can still read and round-trip older on-disk data.

Protocol 0 / release 0.1.38 predates this script; its fixture
(syft_job-0.1.38-protocol0_syftbox) is hand-authored, like protocol-0.json.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml
from syft_job.client import JobClient
from syft_job.config import SyftJobConfig
from syft_job.migrations.registry import JOB_PROTOCOL_VERSION
from syft_job.version import __version__

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"

FIXTURES_DIR = (
    Path(__file__).resolve().parents[1] / "tests" / "migrations" / "p2p" / "fixtures"
)


def _seed_syftbox(syftbox: Path, code_path: Path) -> None:
    """Two jobs: scanned.job (has a review state) and unscanned.job (inbox only)."""
    ds = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DS_EMAIL)
    )
    do = JobClient(
        config=SyftJobConfig(syftbox_folder=syftbox, current_user_email=DO_EMAIL)
    )

    ds.submit_python_job(
        user=DO_EMAIL,
        code_path=str(code_path),
        job_name="scanned.job",
        dependencies=["pandas"],
    )
    do.scan_inbox()  # writes scanned.job's review state.yaml
    # Submitted after the scan, so it has no review state yet.
    ds.submit_bash_job(user=DO_EMAIL, script="echo hello", job_name="unscanned.job")


class _PrettierDumper(yaml.SafeDumper):
    """Indent sequences under their key, matching prettier's YAML style."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _normalize(target: Path) -> None:
    """Strip machine-specific values so the committed fixture is portable.

    The submitter's absolute ``code_path`` and the local syft-client install
    source (an absolute repo path that leaks into config.yaml dependencies and
    run.sh when no DO advertises one) are not part of the release's on-disk
    format; drop the executable bit since fixtures are data, not scripts.
    """
    repo_root = str(Path(__file__).resolve().parents[3])
    for path in target.rglob("*"):
        if not path.is_file():
            continue
        path.chmod(0o644)
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if repo_root in text:
            path.write_text(text.replace(repo_root, "syft-client"))

    for config in target.rglob("config.yaml"):
        data = yaml.safe_load(config.read_text())
        if data.get("code_path"):
            data["code_path"] = f"/tmp/{Path(data['code_path']).name}"
        config.write_text(
            yaml.dump(data, Dumper=_PrettierDumper, default_flow_style=False)
        )


def _build_fixture(target: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        syftbox = tmp_path / "SyftBox"
        syftbox.mkdir()
        code_path = tmp_path / "main.py"
        code_path.write_text('print("hello from job")\n')
        _seed_syftbox(syftbox, code_path)
        shutil.copytree(syftbox, target)
        _normalize(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing fixture"
    )
    args = parser.parse_args()

    target = (
        FIXTURES_DIR / f"syft_job-{__version__}-protocol{JOB_PROTOCOL_VERSION}_syftbox"
    )
    if target.exists():
        if not args.force:
            raise SystemExit(f"{target} already exists; pass --force to regenerate.")
        shutil.rmtree(target)

    _build_fixture(target)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
