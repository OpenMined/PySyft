# The script mainly help in bumping the version of the package
# during release process.

# stdlib
# It covers mainly two scenarios:
# Bumping from beta to the stable version
# Bumping from stable to the next beta version
import argparse
import subprocess
import sys

# Command line processing


def bump_to_stable(version: str):
    print(f"Bumping to stable version {version}.")

    # Invoke bump2version by subprocess and pass the version

    # bumping beta to stable
    subprocess.run(
        [
            "bump2version",
            "prenum",
            "--new-version",
            version,
            "--no-commit",
            "--allow-dirty",
        ],
        check=True,
    )
    # bumping previous stable to current stable
    subprocess.run(
        [
            "bump2version",
            "patch",
            "--config-file",
            ".bumpversion_stable.cfg",
            "--new-version",
            version,
            "--no-commit",
            "--allow-dirty",
        ],
        check=True,
    )


def bump_to_next_beta(version: str):
    version = version + "-beta.0"
    print(f"Bumping to next beta version {version}.")

    # bumping stable to beta
    subprocess.run(
        [
            "bump2version",
            "prenum",
            "--new-version",
            version,
            "--no-commit",
            "--allow-dirty",
        ],
        check=True,
    )


def handle_command_line():
    parser = argparse.ArgumentParser(description="Bump the version of the package.")
    parser.add_argument(
        "--bump-to-stable",
        type=str,
        help="Bump the current beta version to a stable version.",
    )
    parser.add_argument(
        "--bump-to-next-beta",
        type=str,
        help="Bump the current stable version to the next beta version.",
    )

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.bump_to_stable:
        bump_to_stable(args.bump_to_stable)
    elif args.bump_to_next_beta:
        bump_to_next_beta(args.bump_to_next_beta)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    handle_command_line()
