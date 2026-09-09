"""
Auto-approve and run jobs that match specific criteria.

This script runs in an infinite loop, periodically checking for new jobs
and approving/running those that match the specified criteria.
"""

import sys
import time
from pathlib import Path

from syft_rds import login_do
from syft_rds.job_auto_approval import auto_approve_and_run_jobs

# Configuration - edit these values
EMAIL = "your-email@example.com"
TOKEN_PATH = Path("~/.syft/token.json").expanduser()
POLL_INTERVAL = 5  # seconds

# The expected script content
EXPECTED_SCRIPT = """
# Your expected script content here
print("hello")
"""

# Script filename to match
SCRIPT_FILENAME = "main.py"

# Required files - job must contain exactly these files (include the script file)
REQUIRED_FILENAMES = ["main.py", "data.json"]

# Optional: list of allowed user emails (None = allow all)
ALLOWED_USERS = None

# Optional: only allow jobs from approved peers
PEERS_ONLY = False


def main():
    client = login_do(
        email=EMAIL,
        token_path=TOKEN_PATH,
    )

    while True:
        try:
            auto_approve_and_run_jobs(
                client,
                required_file_contents={SCRIPT_FILENAME: EXPECTED_SCRIPT},
                required_file_paths=REQUIRED_FILENAMES,
                allowed_users=ALLOWED_USERS,
                peers_only=PEERS_ONLY,
                verbose=False,
            )
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception:
            pass

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
