"""Test GDriveFilesTransport connection using the actual transport class"""

from pathlib import Path
from google.oauth2.credentials import Credentials
from syft_client.sync.connections.gdrive_transport_v2 import (
    GDriveFilesTransport,
    ProposedFileChange,
    ProposedFileChangesMessage,
)

# Path to credentials
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_PATH = Path(__file__).parent / "token_koen@openmined.org.json"

creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

transport = GDriveFilesTransport(email="koen@openmined.org")
success = transport.setup({"credentials": creds})

msgs = ProposedFileChangesMessage(
    sender_email="koen@openmined.org",
    proposed_file_changes=[
        ProposedFileChange(
            submitted_timestamp=1718544000,
            path="my_file.job",
            content="Hello, world!",
        )
    ],
)

transport.watcher_send_proposed_file_changes_message(
    recipient="koenlennartvanderveen@gmail.com",
    messages=msgs.proposed_file_changes,
)
