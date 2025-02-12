from pathlib import Path

from syftbox.client.base import SyftBoxContextInterface
from syftbox.client.plugins.sync.datasite_state import DatasiteState
from syftbox.client.utils.dir_tree import create_dir_tree
from syftbox.client.utils.display import display_file_tree
from syftbox.lib.ignore import IGNORE_FILENAME, filter_ignored_paths

ignore_file = """
# Exlude alice datasite
/alice@example.com

# exclude all occurrences bob@example.com
bob@example.com

# Exclude all "large" folders under any datasite
*/large/*

# Include important_file.pdf under excluded folder
!/john@example.com/large/important_file.pdf

# General excludes
*.tmp
_.syftignore
*.py[cod]
"""

paths_with_result = [
    # Should be ignored
    ("alice@example.com/file1.txt", True),
    ("john@example.com/results/bob@example.com/file1.txt", True),
    ("john@example.com/large/file1.txt", True),
    ("john@example.com/docs/file1.tmp", True),
    ("script.pyc", True),
    # Should not be ignored
    ("john@example.com/results/alice@example.com/file1.txt", False),
    ("john@example.com/large/important_file.pdf", False),
    ("john@example.com/docs/file3.pdf", False),
    ("script.py", False),
]


def test_ignore_file(tmp_path):
    # without ignore file
    ignore_path = tmp_path / IGNORE_FILENAME
    ignore_path.unlink(missing_ok=True)

    paths, results = zip(*paths_with_result)
    paths = [Path(p) for p in paths]
    filtered_paths = filter_ignored_paths(tmp_path, paths)
    assert filtered_paths == paths

    # with ignore file
    ignore_path.write_text(ignore_file)

    expected_result = [p for p, r in zip(paths, results) if r is False]
    filtered_paths = filter_ignored_paths(tmp_path, paths)
    assert filtered_paths == expected_result


def test_ignore_datasite(datasite_1: SyftBoxContextInterface, datasite_2: SyftBoxContextInterface) -> None:
    datasite_2_files = {
        datasite_2.email: {
            "visible_file.txt": "content",
            "ignored_file.pyc": "content",
        }
    }
    num_files = 2
    num_visible_files = 1
    create_dir_tree(Path(datasite_1.workspace.datasites), datasite_2_files)
    display_file_tree(Path(datasite_1.workspace.datasites))

    # ds1 gets their local state of ds2
    datasite_state = DatasiteState(context=datasite_1, email=datasite_2.email)
    changes = datasite_state.get_datasite_changes()

    assert len(changes.files) == num_visible_files
    assert changes.files[0].path == Path(datasite_2.email) / "visible_file.txt"

    # ds1 ignores ds2
    ignore_path = Path(datasite_1.workspace.datasites) / IGNORE_FILENAME
    with ignore_path.open("a") as f:
        f.write(f"\n/{datasite_2.email}\n")

    # ds1 gets their local state of ds2
    changes = datasite_state.get_datasite_changes()
    assert len(changes.files) == 0

    # remove ignore file
    ignore_path.unlink()
    changes = datasite_state.get_datasite_changes()
    assert len(changes.files) == num_files


def test_ignore_symlinks(datasite_1: SyftBoxContextInterface) -> None:
    # create a symlinked folder containing a file
    folder_to_symlink = Path(datasite_1.workspace.data_dir) / "folder"
    folder_to_symlink.mkdir()
    symlinked_file = folder_to_symlink / "file.txt"
    symlinked_file.write_text("content")

    symlinked_folder = Path(datasite_1.workspace.datasites) / "symlinked_folder"
    symlinked_folder.symlink_to(folder_to_symlink)

    paths = [
        Path("symlinked_folder/file.txt"),
        Path("symlinked_folder/non_existent_file.txt"),
        Path("symlinked_folder/subfolder/file.txt"),
    ]

    filtered_paths = filter_ignored_paths(datasite_1.workspace.datasites, paths, ignore_symlinks=True)
    assert filtered_paths == []


def test_ignore_hidden_files(datasite_1: SyftBoxContextInterface) -> None:
    paths = [
        Path(".hidden_file.txt"),  # Hidden files are filtered
        Path("visible_file.txt"),  # Visible files are not filtered
        Path("subfolder/.hidden_file.txt"),  # Hidden files in folders are filtered
        Path(".subfolder/visible_file.txt"),  # Files in hidden folders are filtered
    ]

    filtered_paths = filter_ignored_paths(datasite_1.workspace.datasites, paths, ignore_hidden_files=True)
    assert filtered_paths == [Path("visible_file.txt")]
