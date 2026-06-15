from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

# Path to credentials

HOME = Path.home()
CLIENT_SECRETS_PATH = HOME / "Downloads" / "credentials.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]


# change this to token_ds.json for your other account
TOKEN_PATH = Path(__file__).parent / "drive_token.json"


def create_token():
    """Create a token for the GDriveFilesTransport"""
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_PATH.absolute(), SCOPES
    )
    creds = flow.run_local_server(port=0)
    with open(TOKEN_PATH, "w") as token:
        token.write(creds.to_json())
    return creds


create_token()
