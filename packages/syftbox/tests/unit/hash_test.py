from pathlib import Path

from syftbox.client.utils.dir_tree import create_dir_tree
from syftbox.lib.hash import collect_files


def test_collect_files(tmp_path: Path):
    # Create entire test structure including symlink targets
    tree = {
        # Symlink targets
        "folder_to_symlink": {"file_in_symlink.txt": "symlink content"},
        "file_to_symlink.txt": "symlink file content",
        # Test structure
        "test_dir": {
            "file1.txt": "content1",
            "file2.txt": "content2",
            ".hidden_file": "hidden",
            ".hidden_dir": {
                "file_in_hidden.txt": "hidden content",
            },
            "nested": {
                "nested_file.txt": "nested content",
            },
        },
    }
    create_dir_tree(tmp_path, tree)

    # Create symlinks in test_dir
    test_dir = tmp_path / "test_dir"
    (test_dir / "symlink_dir").symlink_to(tmp_path / "folder_to_symlink")
    (test_dir / "symlink_file.txt").symlink_to(tmp_path / "file_to_symlink.txt")

    def get_names(files: list[Path]) -> set[str]:
        return {f.name for f in files}

    # Collect excluding hidden and symlinks
    files = collect_files(test_dir)
    assert get_names(files) == {"file1.txt", "file2.txt", "nested_file.txt"}

    # With hidden
    files = collect_files(test_dir, include_hidden=True)
    assert get_names(files) == {
        "file1.txt",
        "file2.txt",
        "nested_file.txt",
        ".hidden_file",
        "file_in_hidden.txt",
    }

    # With symlinks
    files = collect_files(test_dir, follow_symlinks=True)
    assert get_names(files) == {
        "file1.txt",
        "file2.txt",
        "nested_file.txt",
        "file_in_symlink.txt",
        "symlink_file.txt",
    }

    # With both
    files = collect_files(test_dir, include_hidden=True, follow_symlinks=True)
    assert get_names(files) == {
        "file1.txt",
        "file2.txt",
        "nested_file.txt",
        ".hidden_file",
        "file_in_hidden.txt",
        "file_in_symlink.txt",
        "symlink_file.txt",
    }

    # Edge cases
    assert collect_files(test_dir / "nonexistent") == []
    regular_file = test_dir / "just_a_file"
    regular_file.touch()
    assert collect_files(regular_file) == []
