"""Bump a package version and propagate the change to all dependents.

Usage: python scripts/bump_version.py <package-name> <patch|minor|major>

Output (two lines):
  Line 1: new version
  Line 2: space-separated list of all modified pyproject.toml files
"""

import argparse
import re
import tomllib
from pathlib import Path

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
    args = parser.parse_args()

    target_path = find_target_pyproject(args.package_name)
    new_version = update_target_version(target_path, args.bump_type)
    modified_deps = update_dependents(args.package_name, new_version, target_path)

    all_modified = [target_path] + modified_deps
    relative_paths = [str(p.relative_to(REPO_ROOT)) for p in all_modified]

    print(new_version)
    print(" ".join(relative_paths))


if __name__ == "__main__":
    main()
