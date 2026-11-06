"""Dataset source errors must leave the name available for a corrected retry."""

import os
import shutil
from pathlib import Path

import pytest

from syft_datasets.dataset_manager import SyftDatasetManager


MOCK = b"id,value\n1,mock\n"
PRIVATE = b"id,value\n1,private\n"
README = b"# Dataset\n"


def _tree(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): (
            p.read_bytes() if p.is_file() else None,
            p.stat().st_mode,
        )
        for p in root.rglob("*")
    }


def _payload(root: Path, excluded: set[str]) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and p.name not in excluded
    }


@pytest.fixture(params=[["0"], ["1"], ["0", "1"]], ids=["v0", "v1", "both"])
def versions(request):
    return request.param


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    mock, private, readme = (
        source / n for n in ("mock.csv", "private.csv", "README.md")
    )
    for path, data in ((mock, MOCK), (private, PRIVATE), (readme, README)):
        path.write_bytes(data)
    manager = SyftDatasetManager(tmp_path / "SyftBox", "owner@test.org")
    return manager, dict(mock_path=mock, private_path=private, readme_path=readme)


@pytest.mark.parametrize(
    "invalid,exception,message",
    [
        ("private_missing", ValueError, "Private data path"),
        ("readme_missing", FileNotFoundError, "Could not find README"),
        ("readme_directory", FileNotFoundError, "Could not find README"),
        ("readme_extension", ValueError, "markdown"),
        ("mock_missing", FileNotFoundError, "Could not find mock data"),
        pytest.param(
            "private_special",
            ValueError,
            "Private data path",
            marks=pytest.mark.skipif(
                not hasattr(os, "mkfifo"), reason="requires FIFOs"
            ),
        ),
        pytest.param(
            "mock_special",
            ValueError,
            "Mock data path",
            marks=pytest.mark.skipif(
                not hasattr(os, "mkfifo"), reason="requires FIFOs"
            ),
        ),
    ],
)
def test_corrected_source_can_reuse_name(inputs, versions, invalid, exception, message):
    manager, kwargs = inputs
    protected = manager.create(name="protected", protocol_versions=versions, **kwargs)
    protected_before = _tree(protected.mock_dir), _tree(protected.private_dir)
    key = invalid.split("_")[0] + "_path"
    original = kwargs[key]
    original.unlink()
    if invalid == "readme_directory":
        original.mkdir()
    elif invalid == "readme_extension":
        kwargs[key] = original.with_suffix(".txt")
        kwargs[key].write_bytes(README)
    elif invalid.endswith("_special"):
        os.mkfifo(original)

    before = _tree(manager.syftbox_config.syftbox_folder)
    with pytest.raises(exception, match=message):
        manager.create(name="demo", protocol_versions=versions, **kwargs)
    after_error = _tree(manager.syftbox_config.syftbox_folder)

    if original.is_dir():
        original.rmdir()
    elif original.exists():
        original.unlink()
    kwargs[key] = original
    original.write_bytes(
        {"mock_path": MOCK, "private_path": PRIVATE, "readme_path": README}[key]
    )
    created = manager.create(name="demo", protocol_versions=versions, **kwargs)

    assert after_error == before
    assert {p.name: p.read_bytes() for p in created.mock_files} == {"mock.csv": MOCK}
    assert {p.name: p.read_bytes() for p in created.private_files} == {
        "private.csv": PRIVATE
    }
    assert created.readme_path.read_bytes() == README
    assert _tree(protected.mock_dir) == protected_before[0]
    assert _tree(protected.private_dir) == protected_before[1]
    assert manager.get("protected").uid == protected.uid
    refs = list(
        manager.storage.iter_dataset_refs_all_protocols(manager.syftbox_config.email)
    )
    assert sorted(r.protocol_version for r in refs if r.name == "demo") == versions


@pytest.mark.parametrize("kind", ["files", "directories", "empty_directories"])
@pytest.mark.parametrize("readme_name", ["README.md", "README.MD", None])
def test_valid_sources_preserve_payloads(inputs, versions, kind, readme_name):
    manager, kwargs = inputs
    binary = bytes(range(256)) * 257
    if kind == "files":
        kwargs["mock_path"].write_bytes(binary)
        expected_mock, expected_private = {"mock.csv": binary}, {"private.csv": PRIVATE}
    else:
        for key in ("mock_path", "private_path"):
            kwargs[key].unlink()
            kwargs[key].mkdir()
        expected_mock, expected_private = {}, {}
        if kind == "directories":
            for key in ("mock_path", "private_path"):
                folder = kwargs[key] / "nested"
                folder.mkdir()
                (folder / "payload.bin").write_bytes(binary)
                (kwargs[key] / "empty.bin").write_bytes(b"")
            expected_mock = expected_private = {
                "nested/payload.bin": binary,
                "empty.bin": b"",
            }
    if readme_name is None:
        kwargs["readme_path"] = None
    else:
        path = kwargs["readme_path"].with_name(readme_name)
        kwargs["readme_path"].rename(path)
        kwargs["readme_path"] = path
    source_before = _tree(kwargs["mock_path"].parent)

    created = manager.create_all(
        name="demo", protocol_versions=versions, users=["scientist@test.org"], **kwargs
    )

    assert sorted(created) == versions
    assert len({d.uid for d in created.values()}) == 1
    for dataset in created.values():
        excluded = {"dataset.yaml", "private_metadata.yaml", "syft.pub.yaml"}
        if readme_name is not None:
            excluded.add(readme_name)
            assert dataset.readme_path.read_bytes() == README
        else:
            assert dataset.readme_path is None
        assert _payload(dataset.mock_dir, excluded) == expected_mock
        assert _payload(dataset.private_dir, excluded) == expected_private
        assert {
            p.relative_to(dataset.mock_dir).as_posix(): p.read_bytes()
            for p in dataset.mock_files
        } == expected_mock
    assert _tree(kwargs["mock_path"].parent) == source_before


def test_duplicate_keeps_existing_dataset(inputs, versions):
    manager, kwargs = inputs
    dataset = manager.create(name="demo", protocol_versions=versions, **kwargs)
    before = _tree(manager.syftbox_config.syftbox_folder)

    with pytest.raises(FileExistsError, match="already exists"):
        manager.create(name="demo", protocol_versions=versions, **kwargs)

    assert _tree(manager.syftbox_config.syftbox_folder) == before
    assert manager.get("demo").uid == dataset.uid


def test_reserved_metadata_rejected_without_copying(inputs, versions):
    manager, kwargs = inputs
    mock = kwargs["mock_path"]
    mock.unlink()
    mock.mkdir()
    (mock / "dataset.yaml").write_bytes(b"reserved\n")

    with pytest.raises(ValueError, match="reserved file"):
        manager.create(name="demo", protocol_versions=versions, **kwargs)

    assert not manager.syftbox_config.syftbox_folder.exists()


def test_invalid_source_preserves_existing_empty_targets(inputs, versions):
    manager, kwargs = inputs
    for version in versions:
        ref = manager.storage.new_dataset_ref("demo", version)
        manager.storage.public_dataset_dir(ref).mkdir(parents=True, mode=0o750)
        manager.storage.private_dataset_dir(ref).mkdir(parents=True, mode=0o700)
    kwargs["private_path"].unlink()
    before = _tree(manager.syftbox_config.syftbox_folder)

    with pytest.raises(ValueError, match="Private data path"):
        manager.create(name="demo", protocol_versions=versions, **kwargs)

    assert _tree(manager.syftbox_config.syftbox_folder) == before
    kwargs["private_path"].write_bytes(PRIVATE)
    dataset = manager.create(name="demo", protocol_versions=versions, **kwargs)
    assert {p.name: p.read_bytes() for p in dataset.mock_files} == {"mock.csv": MOCK}
    assert {p.name: p.read_bytes() for p in dataset.private_files} == {
        "private.csv": PRIVATE
    }


@pytest.mark.parametrize("changed", ["private", "readme"])
def test_source_removed_during_copy_is_rejected(inputs, versions, monkeypatch, changed):
    manager, kwargs = inputs
    protected = manager.create(name="protected", protocol_versions=versions, **kwargs)
    protected_before = _tree(manager.syftbox_config.syftbox_folder)
    copy_file = shutil.copy2
    source = kwargs[changed + "_path"]

    def copy_then_remove(src, dst, *args, **copy_options):
        result = copy_file(src, dst, *args, **copy_options)
        if Path(src) == kwargs["mock_path"]:
            source.unlink()
        return result

    monkeypatch.setattr(shutil, "copy2", copy_then_remove)
    exception = ValueError if changed == "private" else FileNotFoundError
    message = "Private data path" if changed == "private" else "Could not find README"

    with pytest.raises(exception, match=message):
        manager.create(name="demo", protocol_versions=versions, **kwargs)

    assert [d.name for d in manager.get_all()] == ["protected"]
    assert manager.get("protected").uid == protected.uid
    after = _tree(manager.syftbox_config.syftbox_folder)
    for path, value in protected_before.items():
        assert after[path] == value
    for version in versions:
        ref = manager.storage.new_dataset_ref("demo", version)
        assert not manager.storage.metadata_path(ref).exists()
        assert not manager.storage.private_metadata_path(ref).exists()
