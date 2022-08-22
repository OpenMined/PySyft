from getpass import getpass
from typing import Optional
from oblv import OblvClient

def user_public_key(client: OblvClient):
    return client.psk()