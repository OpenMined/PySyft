# stdlib
from getpass import getpass
from typing import Optional

# third party
from oblv import authenticate


def login(apikey: Optional[str] = None):
    if apikey is None:
        apikey = getpass("Please provide your oblv API_KEY to login:")

    return authenticate(apikey)
