"""JSON requests to the Firebase REST APIs."""

import json
import urllib.error
import urllib.request
from typing import Any

from syft.sync.connections.exceptions import ConnectionUnavailableError

REQUEST_TIMEOUT_SECONDS = 20


class FirebaseRequestError(ConnectionError):
    """A Firebase REST API refused a request."""

    def __init__(self, url: str, status: int, reason: str):
        super().__init__(f"{url} answered {status}: {reason}")
        self.status = status


def post_json(url: str, body: Any, id_token: str | None = None) -> Any:
    """POST ``body`` as JSON, as the user of ``id_token`` if one is given, and
    return the JSON answer.

    Raises:
        FirebaseRequestError: The API answers with an error status.
        ConnectionUnavailableError: The API cannot be reached.
    """
    headers = {"Content-Type": "application/json"}
    if id_token is not None:
        headers["Authorization"] = f"Bearer {id_token}"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    return send_request(request)


def request_json(method: str, url: str, body: Any = None) -> Any:
    """Send ``body`` as JSON with ``method``, and return the JSON answer.

    The credentials, if any, are in the query of ``url``.

    Raises:
        FirebaseRequestError: The API answers with an error status.
        ConnectionUnavailableError: The API cannot be reached.
    """
    data = None if body is None else json.dumps(body).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    return send_request(request)


def get_json(url: str, id_token: str) -> Any:
    """GET ``url`` as the user of ``id_token``, and return the JSON answer.

    Raises:
        FirebaseRequestError: The API answers with an error status.
        ConnectionUnavailableError: The API cannot be reached.
    """
    headers = {"Authorization": f"Bearer {id_token}"}
    return send_request(urllib.request.Request(url, headers=headers))


def send_request(request: urllib.request.Request) -> Any:
    # The query can hold an ID token, so no message shows it.
    url = without_query(request.full_url)
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        reason = err.read().decode(errors="replace")
        raise FirebaseRequestError(url, err.code, reason) from err
    except OSError as err:
        raise ConnectionUnavailableError(f"Could not reach {url}: {err}") from err


def without_query(url: str) -> str:
    return url.split("?", 1)[0]
