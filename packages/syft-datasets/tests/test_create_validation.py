"""A create with a bad source path copies nothing, so a retry can reuse the name."""

from pathlib import Path

import pytest

from syft_datasets.dataset_manager import SyftDatasetManager


@pytest.mark.parametrize(
    "key,bad_name,error",
    [
        ("private_path", "missing.csv", ValueError),
        ("readme_path", "missing.md", FileNotFoundError),
        ("readme_path", "README.txt", ValueError),
    ],
)
def test_bad_source_copies_nothing_and_retry_works(
    tmp_path: Path, key: str, bad_name: str, error: type[Exception]
):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("mock.csv", "private.csv", "README.md", "README.txt"):
        (source / name).write_text(name)
    paths = {
        "mock_path": source / "mock.csv",
        "private_path": source / "private.csv",
        "readme_path": source / "README.md",
    }
    syftbox = tmp_path / "SyftBox"
    manager = SyftDatasetManager(syftbox, "owner@test.org")

    with pytest.raises(error):
        manager.create(
            name="demo",
            protocol_versions=["0", "1"],
            **{**paths, key: source / bad_name},
        )
    assert not syftbox.exists()

    dataset = manager.create(name="demo", protocol_versions=["0", "1"], **paths)
    assert [f.name for f in dataset.private_files] == ["private.csv"]
