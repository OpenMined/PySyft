import requests

from syft.grid.exceptions import GridError


def _send_http_req(url, method, path: str, params: dict = None, body: bytes = None):
    if method == "GET":
        res = requests.get(url + path, params)
    elif method == "POST":
        res = requests.post(url + path, params=params, data=body)

    if not res.ok:
        raise GridError("HTTP response is not OK", res.status_code)

    response = res.content
    return response
