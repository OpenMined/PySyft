# stdlib
from distutils.dir_util import copy_tree
from pathlib import Path

COPY_SUFFIX = "-copy"


def make_copy_path(path: Path):
    return f"{path.parent}/{path.stem}{COPY_SUFFIX}{path.suffix}"


def make_original_path(copy_path: Path):
    return f"{str(copy_path)[:-len(COPY_SUFFIX)]}"


def make_copy(server):
    cfg = server.python_server.db_config
    original_dir = cfg.path.resolve()
    copy_dir = f"{original_dir}{COPY_SUFFIX}"
    copy_tree(original_dir, copy_dir)
    print(f"moved\n{original_dir}\nto\n{copy_dir}\n")


def restore_copy(copy_dir):
    copy_dir = Path(copy_dir)
    original_dir = make_original_path(copy_dir)
    copy_tree(copy_dir, original_dir)
    print(f"moved\n{copy_dir}\nto\n{original_dir}\n")
