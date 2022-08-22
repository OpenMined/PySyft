from getpass import getpass
from typing import Optional
from oblv import authenticate

def login(apikey : Optional[str] = None):
    if apikey==None:
        apikey = getpass("Please provide your oblv apikey to login:")
    try:
        return authenticate(apikey)
    except Exception as e:
        print("Failed to Authenticate with message : ")
        print(e)
        raise(e)