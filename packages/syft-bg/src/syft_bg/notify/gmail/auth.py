"""Gmail OAuth authentication."""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/pubsub",
]


def run_oauth_flow_manual(flow: InstalledAppFlow) -> Credentials:
    """Run OAuth flow manually without browser auto-launch.

    This works in headless environments like Colab, SSH, and containers.
    """
    # Set redirect URI for out-of-band (manual) flow
    # Using loopback address - user will get a "can't connect" page but can copy the code from URL
    flow.redirect_uri = "http://localhost:1"

    # Generate authorization URL
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")

    print()
    print("Please visit this URL to authorize the application:")
    print()
    print(f"    {auth_url}")
    print()
    print("After authorizing, you'll be redirected to a page that won't load.")
    print("Copy the 'code' parameter from the URL in your browser's address bar.")
    print("The URL will look like: http://localhost:1/?code=XXXXX&scope=...")
    print()

    # Get the authorization code from user
    code = input("Enter the authorization code: ").strip()

    # Exchange code for credentials
    flow.fetch_token(code=code)
    return flow.credentials


def authenticate_and_save(creds_path: Path, token_path: Path) -> bool:
    """Run Gmail OAuth flow and save the token to disk.

    Returns True if authentication succeeded, False otherwise.
    """
    print("Setting up Gmail authentication...")
    print("This is needed for email notifications.\n")
    try:
        auth = GmailAuth()
        credentials = auth.authenticate_user(creds_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json())
        print(f"Gmail token saved to {token_path}\n")
        return True
    except Exception as e:
        print(f"Gmail setup failed: {e}\n")
        return False


class GmailAuth:
    """Handles Gmail OAuth authentication."""

    def authenticate_user(self, credentials_path: Path) -> Credentials:
        """Run OAuth flow to get Gmail credentials."""
        credentials_path = Path(credentials_path).expanduser()
        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path), GMAIL_SCOPES
        )
        return run_oauth_flow_manual(flow)

    def load_credentials(self, token_path: Path) -> Credentials:
        """Load credentials from token file, refreshing if needed.

        Loads without restricting scopes so a shared token file
        (e.g. one that also carries Drive scopes) is not narrowed
        on refresh.
        """
        token_path = Path(token_path).expanduser()
        creds = Credentials.from_authorized_user_file(str(token_path))

        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, "w") as f:
                f.write(creds.to_json())

        return creds
