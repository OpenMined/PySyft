from pathlib import Path


def authenticate_drive(drive_token_path: Path, creds_path: Path):
    print("\nSetting up Google Drive authentication...")
    print("This is needed for monitoring jobs and peers.\n")
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow

        from syft_bg.common.drive import DRIVE_SCOPES

        flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), DRIVE_SCOPES)
        flow.redirect_uri = "http://localhost:1"
        auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")

        print("Visit this URL to authorize Google Drive access:\n")
        print(f"  {auth_url}\n")
        print("After authorizing, you'll see a page that won't load.")
        print("Copy the 'code' value from the URL in your browser's address bar.")
        print("(The URL looks like: http://localhost:1/?code=XXXXX&scope=...)\n")

        code = input("Paste the authorization code here: ").strip()
        flow.fetch_token(code=code)
        creds = flow.credentials

        drive_token_path.parent.mkdir(parents=True, exist_ok=True)
        drive_token_path.write_text(creds.to_json())
        print(f"\nDrive token saved to {drive_token_path}")
    except Exception as e:
        print(f"Drive setup failed: {e}")
