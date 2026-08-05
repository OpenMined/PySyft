"""Bump the version of one package, and update the packages that depend on it.

Usage:
    python scripts/bump_version.py <package-name> <patch|minor|major>
                                   [--dependents {bumped,published}]

The script writes the new version into the pyproject.toml of the package. It
then writes a version pin for the package into each pyproject.toml that depends
on it.

The --dependents option selects the version for those pins:

- published: the version that was in the file before this run. A release
  publishes the version on the branch, and bumps the version after that. This
  version is therefore the version on PyPI. Use this option for a release.
- bumped: the new version. PyPI does not have this version yet. Use this option
  only if the script runs before the release.

The script prints two lines:

- Line 1: the new version.
- Line 2: the modified pyproject.toml files, separated by spaces.
"""

import argparse
import re
from pathlib import Path

import tomllib
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_all_pyproject_files() -> list[Path]:
    """Find root and all packages/*/pyproject.toml files."""
    files = []
    root = REPO_ROOT / "pyproject.toml"
    if root.exists():
        files.append(root)
    for p in sorted((REPO_ROOT / "packages").glob("*/pyproject.toml")):
        files.append(p)
    return files


def find_target_pyproject(package_name: str) -> Path:
    """Find the pyproject.toml whose project.name matches package_name."""
    for path in find_all_pyproject_files():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        if data.get("project", {}).get("name") == package_name:
            return path
    raise SystemExit(f"Package '{package_name}' not found in any pyproject.toml")


def bump_version(current: Version, bump_type: str) -> Version:
    if bump_type == "major":
        return Version(f"{current.major + 1}.0.0")
    elif bump_type == "minor":
        return Version(f"{current.major}.{current.minor + 1}.0")
    else:
        return Version(f"{current.major}.{current.minor}.{current.micro + 1}")


def update_target_version(path: Path, bump_type: str) -> Version:
    """Bump the version field in the target pyproject.toml. Returns new version."""
    with open(path, "rb") as f:
        current = Version(tomllib.load(f)["project"]["version"])
    new = bump_version(current, bump_type)
    text = path.read_text()
    text = text.replace(f'version = "{current}"', f'version = "{new}"')
    path.write_text(text)
    return new


def update_dependents(
    package_name: str, new_version: Version, target_path: Path
) -> list[Path]:
    """Update all pyproject.toml files that depend on package_name. Returns modified paths."""
    modified = []
    pattern = re.compile(
        rf'"({re.escape(package_name)})'  # package name
        rf'(?:[><=!]=?[^"]*)?'  # optional version specifier
        rf'"'  # closing quote
        rf"(?!\s*=)"  # not a TOML key (e.g. in [tool.uv.sources])
    )
    replacement = f'"{package_name}=={new_version}"'

    for path in find_all_pyproject_files():
        if path == target_path:
            continue
        text = path.read_text()
        new_text = pattern.sub(replacement, text)
        if new_text != text:
            path.write_text(new_text)
            modified.append(path)
    return modified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bump package version with dependency propagation"
    )
    parser.add_argument("package_name", help="Package name (e.g. syft-perms)")
    parser.add_argument("bump_type", choices=["major", "minor", "patch"])
    parser.add_argument(
        "--dependents",
        choices=["bumped", "published"],
        default="bumped",
        help=(
            "Version for the dependent pins. 'bumped' is the new version. "
            "'published' is the version that was in the file before this run, "
            "which is the version a release publishes."
        ),
    )
    args = parser.parse_args()

    target_path = find_target_pyproject(args.package_name)
    with open(target_path, "rb") as f:
        published_version = Version(tomllib.load(f)["project"]["version"])
    new_version = update_target_version(target_path, args.bump_type)
    pinned = new_version if args.dependents == "bumped" else published_version
    modified_deps = update_dependents(args.package_name, pinned, target_path)

    all_modified = [target_path] + modified_deps
    relative_paths = [str(p.relative_to(REPO_ROOT)) for p in all_modified]

    print(new_version)
    print(" ".join(relative_paths))


if __name__ == "__main__":
    main()
