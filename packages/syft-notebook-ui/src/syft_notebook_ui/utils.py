from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.tree import Tree

from syft_notebook_ui.types import TableDict, TableList

PERM_FILE = "syftperm.yaml"


def display(obj: Any) -> Any:
    """Convert an object to a displayable object, if possible."""

    if isinstance(obj, list):
        return TableList(obj)
    elif isinstance(obj, dict):
        return TableDict(obj)

    return obj


def make_dirtree_string(root_dir: Path) -> Optional[str]:
    try:
        # Create a Tree object
        tree = Tree(f"📁 {root_dir.name}")

        def add_dir(tree: Tree, path: Path) -> None:
            for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
                if child.is_dir():
                    sub_tree = tree.add(f"📁 {child.name}")
                    add_dir(sub_tree, child)
                elif child.name == PERM_FILE:
                    tree.add(f"🛡️ {child.name}")
                else:
                    tree.add(f"📄 {child.name}")

        add_dir(tree, root_dir)
        console = Console()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()
    except Exception as e:
        return f"Could not generate directory tree: {e}"


def show_dir(path: Path) -> None:
    """
    Display the contents of a directory in a rich tree format.

    Args:
        path (Path): The path to the directory to display.
    """
    tree_str = make_dirtree_string(path)
    if tree_str:
        print(tree_str)
    else:
        print(f"Could not display directory: {path}")
