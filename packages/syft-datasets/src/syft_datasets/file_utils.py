import shutil
from pathlib import Path
from typing import Iterator


def copy_dir_contents(src: Path, dst: Path, exists_ok: bool = False) -> list[Path]:
    if not src.is_dir():
        raise ValueError(f"Source path {src} is not a directory.")
    return copy_paths(src.iterdir(), dst, exists_ok)


def copy_paths(files: Iterator[Path], dst: Path, exists_ok: bool = False) -> list[Path]:
    """
    Copy a list of files to a destination directory.
    If `dst` does not exist, it will be created.

    Returns:
        list[Path]: List of absolute paths to all copied files (destination paths).
    """
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for file in files:
        dst_path = dst / file.name
        if dst_path.exists() and not exists_ok:
            raise FileExistsError(f"Destination path {dst_path} already exists.")
        if file.is_file():
            shutil.copy2(file, dst_path)
            copied_files.append(dst_path.absolute())
        elif file.is_dir():
            shutil.copytree(file, dst_path, dirs_exist_ok=exists_ok)
            # Get all files in the copied directory tree
            for copied_file in dst_path.rglob("*"):
                if copied_file.is_file():
                    copied_files.append(copied_file.absolute())

    return copied_files


def is_empty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return not any(path.iterdir())
