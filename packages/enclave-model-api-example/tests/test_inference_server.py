"""Unit tests for the inference service pieces: paths, log writer, FastAPI app."""

import ast
import json
from pathlib import Path

from fastapi.testclient import TestClient

from enclave_model_api.log_writer import (
    LOG_FILE_NAME,
    append_log_record,
    build_log_record,
)
from enclave_model_api.paths import (
    candidate_private_dataset_dirs,
    private_dataset_dir,
    resolve_weights_dir,
    weights_ready,
)
from enclave_model_api.server import create_app
from enclave_model_api.service import InferenceService
from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_storage import DatasetStorage
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION

from inference_stub import STUB_COMPLETION_PREFIX, StubBackend, make_stub_weights


def test_log_writer_appends_jsonl(tmp_path):
    logs_dir = tmp_path / "logs"

    record = build_log_record("a prompt", "a completion", {"elapsed": 0.5})
    log_file = append_log_record(logs_dir, record)
    append_log_record(logs_dir, build_log_record("p2", "c2", {"elapsed": 0.1}))

    assert log_file == logs_dir / LOG_FILE_NAME
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["prompt"] == "a prompt"
    assert lines[0]["completion"] == "a completion"
    assert {"id", "timestamp", "prompt", "completion", "stats"} <= lines[0].keys()


def test_weights_ready(tmp_path):
    weights_dir = tmp_path / "weights"
    assert not weights_ready(weights_dir)

    weights_dir.mkdir()
    (weights_dir / "tokenizer.model").write_bytes(b"x")
    assert not weights_ready(weights_dir)  # no checkpoint dir yet

    make_stub_weights(weights_dir)
    assert weights_ready(weights_dir)


def test_private_dataset_dir_layout(tmp_path):
    # A dataset not yet on disk falls back to the current protocol, which is the
    # layout this client writes for a dataset of its own (the enclave's logs).
    # A dataset another datasite writes may arrive in an older layout, and
    # candidate_private_dataset_dirs covers that case.
    path = private_dataset_dir(tmp_path, "enclave@openmined.org", "inference_logs")
    assert path == (
        tmp_path
        / "enclave@openmined.org"
        / "private"
        / "syft_datasets"
        / f"v{DATASET_PROTOCOL_VERSION}"
        / "inference_logs"
    )


def test_private_dataset_dir_follows_the_layout_on_disk(tmp_path):
    # A dataset already on disk decides its own layout, whatever the fallback is.
    flat = (
        tmp_path
        / "enclave@openmined.org"
        / "private"
        / "syft_datasets"
        / "inference_logs"
    )
    flat.mkdir(parents=True)
    (flat / "private_metadata.yaml").write_text("uid: x\n")
    public = (
        tmp_path
        / "enclave@openmined.org"
        / "public"
        / "syft_datasets"
        / "inference_logs"
    )
    public.mkdir(parents=True)
    (public / "dataset.yaml").write_text("name: inference_logs\n")

    path = private_dataset_dir(tmp_path, "enclave@openmined.org", "inference_logs")
    assert path == flat


def test_inference_server_full_lifecycle(tmp_path):
    """503 before weights → load after weights sync → /infer logs each request."""
    weights_dir = tmp_path / "weights"
    logs_dir = tmp_path / "logs"
    service = InferenceService(
        backend=StubBackend(),
        model_size="270m",
        weights_dir=weights_dir,
        logs_dir=logs_dir,
    )
    client = TestClient(create_app(service, use_encryption=True))

    # Weights not synced yet
    status = client.get("/model-status").json()
    assert status == {
        "model_size": "270m",
        "mock": False,
        "weights_present": False,
        "model_loaded": False,
        "use_encryption": True,
    }
    assert client.post("/infer", json={"query": "hi"}).status_code == 503

    # Weights arrive (as if synced from the model owner) and get loaded
    make_stub_weights(weights_dir)
    assert service.try_load()

    response = client.post("/infer", json={"query": "What is the capital of NL?"})
    assert response.status_code == 200
    body = response.json()
    assert body["completion"].startswith(STUB_COMPLETION_PREFIX)
    assert body["stats"]["elapsed"] > 0

    status = client.get("/model-status").json()
    assert status["weights_present"] and status["model_loaded"]

    records = [
        json.loads(line) for line in (logs_dir / LOG_FILE_NAME).read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["prompt"] == "What is the capital of NL?"
    assert records[0]["completion"] == body["completion"]


def _storage(tmp_path, email: str) -> DatasetStorage:
    return DatasetStorage(config=SyftBoxConfig(syftbox_folder=tmp_path, email=email))


def _write_weights(private_dir):
    private_dir.mkdir(parents=True, exist_ok=True)
    make_stub_weights(private_dir)


def test_candidate_dirs_cover_every_layout_while_the_dataset_is_absent(tmp_path):
    # The owner writes the layout its own release decided, so a reader that is
    # still waiting cannot know which one arrives.
    storage = _storage(tmp_path, "owner@test.org")
    candidates = candidate_private_dataset_dirs(storage, "owner@test.org", "weights")

    assert [c.name for c in candidates] == ["weights"] * len(candidates)
    segments = [c.parent.name for c in candidates]
    # Newest layout first, down to the floor.
    assert segments[0] == f"v{DATASET_PROTOCOL_VERSION}"
    assert "syft_datasets" in segments


def test_candidate_dirs_follow_the_layout_on_disk(tmp_path):
    storage = _storage(tmp_path, "owner@test.org")
    flat_private = tmp_path / "owner@test.org" / "private" / "syft_datasets" / "weights"
    flat_private.mkdir(parents=True)
    (flat_private / "private_metadata.yaml").write_text("uid: x\n")
    flat_public = tmp_path / "owner@test.org" / "public" / "syft_datasets" / "weights"
    flat_public.mkdir(parents=True)
    (flat_public / "dataset.yaml").write_text("name: weights\n")

    assert candidate_private_dataset_dirs(storage, "owner@test.org", "weights") == [
        flat_private
    ]


def test_resolve_weights_dir_finds_the_flat_layout_of_an_earlier_release(tmp_path):
    # Every data owner in the fleet today writes protocol 0, so the weights land
    # flat. A guess fixed at the current layout never sees them.
    owner = "owner@test.org"
    flat = tmp_path / owner / "private" / "syft_datasets" / "weights"
    _write_weights(flat)

    assert resolve_weights_dir(tmp_path, owner, "weights") == flat
    assert weights_ready(resolve_weights_dir(tmp_path, owner, "weights"))


def test_resolve_weights_dir_returns_the_newest_candidate_while_absent(tmp_path):
    owner = "owner@test.org"
    resolved = resolve_weights_dir(tmp_path, owner, "weights")
    assert resolved.parent.name == f"v{DATASET_PROTOCOL_VERSION}"
    assert not weights_ready(resolved)


def test_the_service_sees_weights_that_land_in_an_older_layout(tmp_path):
    # The poll exists so the enclave need not restart. It re-resolves the
    # layout, so weights arriving flat after startup are picked up.
    owner = "owner@test.org"
    service = InferenceService(
        backend=StubBackend(),
        model_size="270m",
        weights_dir=lambda: resolve_weights_dir(tmp_path, owner, "weights"),
        logs_dir=tmp_path / "logs",
    )
    assert not service.weights_present

    _write_weights(tmp_path / owner / "private" / "syft_datasets" / "weights")

    assert service.weights_present
    assert service.try_load()


# --- entrypoint wiring ----------------------------------------------------
#
# Neither entrypoint can be imported by a test: each starts a server at import
# time. The invariant they must hold is checked on their syntax tree instead.
# A path fixed at startup defeats the poll, and both entrypoints have to pass a
# callable that re-resolves the layout.


def _entrypoints():
    root = Path(__file__).resolve().parents[1]
    return sorted((root / "scripts").glob("*.py")) + sorted(
        (root / "docker").glob("inference_server.py")
    )


def _weights_argument(path: Path):
    """The ``weights_dir`` argument of the InferenceService call in a module."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "InferenceService"
        ):
            for keyword in node.keywords:
                if keyword.arg == "weights_dir":
                    return keyword.value
    return None


def test_every_entrypoint_re_resolves_the_weights_layout():
    checked = []
    for path in _entrypoints():
        argument = _weights_argument(path)
        if argument is None:
            continue
        checked.append(path.name)
        assert isinstance(argument, ast.Lambda), (
            f"{path.name} fixes the weights path at startup; the owner may write "
            "an older layout, and the poll would never see it"
        )
        called = {
            n.func.id
            for n in ast.walk(argument)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "resolve_weights_dir" in called, (
            f"{path.name} must resolve the weights layout through "
            "resolve_weights_dir, which probes every layout it supports"
        )
    # Both entrypoints, or the glob stopped matching.
    assert sorted(checked) == ["inference_server.py", "local_server.py"]
