# stdlib
from getpass import getpass
from typing import Any

# third party
from oblv_ctl import authenticate


def login(apikey: str | None = None) -> Any:
    if apikey is None:
        apikey = getpass("Please provide your oblv API_KEY to login:")

    return authenticate(apikey)
