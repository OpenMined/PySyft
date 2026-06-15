"""Test GDriveFilesTransport connection using the actual transport class"""

from pprint import pprint
from pathlib import Path
from google.oauth2.credentials import Credentials
from syft_client.sync.connections.gdrive_transport_v2 import GDriveFilesTransport

# Path to credentials
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_PATH = Path(__file__).parent / "token_koenlennartvanderveen@gmail.com.json"

creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

transport = GDriveFilesTransport(email="koenlennartvanderveen@gmail.com")
success = transport.setup({"credentials": creds})
assert success, "Failed to setup transport"

msgs = transport._get_messages_from_transport(sender_email="koen@openmined.org")

pprint(msgs)
