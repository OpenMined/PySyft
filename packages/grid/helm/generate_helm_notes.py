# stdlib
import json
import os
from pathlib import Path
import sys


def add_notes(helm_chart_template_dir: str) -> None:
    """Add notes or information post helm install or upgrade."""

    notes = """
    Thank you for installing {{ .Chart.Name }}.
    Your release is named {{ .Release.Name }}.
    To learn more about the release, try:

        $ helm status {{ .Release.Name }} -n {{ .Release.Namespace }}
        $ helm get all {{ .Release.Name }}
    """

    notes_path = os.path.join(helm_chart_template_dir, "NOTES.txt")

    protocol_changelog = get_protocol_changes()

    notes += "\n" + protocol_changelog

    with open(notes_path, "w") as fp:
        fp.write(notes)


def get_protocol_changes() -> str:
    """Generate change log of the dev protocol state."""
    script_path = os.path.dirname(os.path.realpath(__file__))
    protocol_path = Path(
        os.path.normpath(
            os.path.join(
                script_path,
                "../../",
                "syft/src/syft/protocol",
                "protocol_version.json",
            )
        )
    )

    protocol_changes = ""
    if protocol_path.exists():
        dev_protocol_changes = json.loads(protocol_path.read_text()).get("dev", {})
        protocol_changes = json.dumps(
            dev_protocol_changes.get("object_versions", {}), indent=4
        )

    protocol_changelog = f"""
    Following class versions are either added/removed.

    {protocol_changes}

    This means the existing data will be automatically be migrated to
    their latest class versions during the upgrade.
    """

    return protocol_changelog


if __name__ == "__main__":
    # write code to path from user and pass to generate notes
    if len(sys.argv) != 2:
        print("Please provide helm chart template directory path")
        sys.exit(1)
    helm_chart_template_dir = sys.argv[1]
    add_notes(helm_chart_template_dir)
    print("=" * 50)
    print("Notes Generated Successfully")
    print("=" * 50)
