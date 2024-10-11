# stdlib
from distutils.dir_util import copy_tree
from pathlib import Path

# relative
from ..orchestra import ServerHandle

COPY_SUFFIX = "-copy"


def make_copy_path(path: Path) -> str:
    return f"{path.parent}/{path.stem}{COPY_SUFFIX}{path.suffix}"


def make_original_path(copy_path: Path) -> str:
    return f"{str(copy_path)[:-len(COPY_SUFFIX)]}"


def make_copy(server: ServerHandle) -> None:
    if not server.python_server:
        print("server does not have python server, no copy made")
        return
    cfg = server.python_server.db_config
    original_dir = str(cfg.path.resolve())
    copy_dir = f"{original_dir}{COPY_SUFFIX}"
    copy_tree(original_dir, copy_dir)
    print(f"copied\n{original_dir}\nto\n{copy_dir}\n")


def restore_copy(copy_dir: str) -> None:
    copy_dir_path = Path(copy_dir)
    original_dir = make_original_path(copy_dir_path)
    copy_tree(copy_dir_path, original_dir)
    print(f"copied\n{copy_dir}\nto\n{original_dir}\n")
