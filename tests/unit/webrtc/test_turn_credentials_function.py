"""The reply of the TURN Cloud Function to a Firebase ID token that it cannot
verify. Only the function's own environment has the Firebase libraries, so the
test runs the function with stand-ins for them."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

FUNCTIONS_FOLDER = Path(__file__).parents[3] / "infra" / "firebase" / "functions"
# verify_id_token raises a ValueError like this one when the function's
# project is not set up.
SERVER_DETAIL = "Failed to ascertain project ID from the credential"


class InvalidIdTokenError(Exception):
    pass


def refuse_token(token: str) -> dict:
    raise ValueError(SERVER_DETAIL)


class Request:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


class Response:
    def __init__(self, body: str, status: int, content_type: str):
        self.body = body
        self.status = status


class SecretParam:
    def __init__(self, name: str):
        self.name = name


def stand_in_module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture
def turn_credentials(monkeypatch):
    """The function in main.py, with stand-ins for the Firebase libraries."""
    auth = stand_in_module(
        "firebase_admin.auth",
        verify_id_token=refuse_token,
        InvalidIdTokenError=InvalidIdTokenError,
    )
    https_fn = stand_in_module(
        "firebase_functions.https_fn",
        on_request=lambda **options: lambda function: function,
        Request=Request,
        Response=Response,
    )
    stand_ins = [
        stand_in_module("firebase_admin", auth=auth, initialize_app=lambda: None),
        stand_in_module("firebase_functions", https_fn=https_fn),
        stand_in_module("firebase_functions.params", SecretParam=SecretParam),
    ]
    for module in stand_ins:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.syspath_prepend(str(FUNCTIONS_FOLDER))
    spec = importlib.util.spec_from_file_location(
        "turn_function", FUNCTIONS_FOLDER / "main.py"
    )
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    return main.turn_credentials


def test_refused_token_reply_hides_exception_text(turn_credentials):
    response = turn_credentials(Request({"Authorization": "Bearer bad-token"}))

    assert response.status == 401
    assert SERVER_DETAIL not in json.loads(response.body)["error"]
