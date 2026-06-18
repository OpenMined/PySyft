"""Unit tests for the inference service pieces: paths, log writer, FastAPI app."""

import json

from fastapi.testclient import TestClient

from enclave_model_api.log_writer import (
    LOG_FILE_NAME,
    append_log_record,
    build_log_record,
)
from enclave_model_api.paths import private_dataset_dir, weights_ready
from enclave_model_api.server import create_app
from enclave_model_api.service import InferenceService

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
    path = private_dataset_dir(tmp_path, "enclave@openmined.org", "inference_logs")
    assert path == (
        tmp_path
        / "enclave@openmined.org"
        / "private"
        / "syft_datasets"
        / "inference_logs"
    )


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
    client = TestClient(create_app(service))

    # Weights not synced yet
    status = client.get("/model-status").json()
    assert status == {
        "model_size": "270m",
        "weights_present": False,
        "model_loaded": False,
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
