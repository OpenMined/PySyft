"""
Auto-approve and run jobs that match specific criteria.

This script runs in an infinite loop, periodically checking for new jobs
and approving/running those that match the specified criteria.
"""

import sys
import time
from pathlib import Path

from syft_rds import login_do
from syft_rds.job_auto_approval import (
    auto_approve_and_run_jobs,
    validate_criteria_paths,
)

# Configuration - edit these values
EMAIL = "your-email@example.com"
TOKEN_PATH = Path("~/.syft/token.json").expanduser()
POLL_INTERVAL = 5  # seconds

# The expected script content
EXPECTED_SCRIPT = """
# Your expected script content here
print("hello")
"""

# Script path to match, relative to the job submission root
SCRIPT_PATH = "code/main.py"

# The script the runner executes. Take it from a job you have reviewed:
# `client.jobs[0].run_script`. A job with any other run.sh is not approved.
EXPECTED_RUN_SCRIPT = """
# The reviewed run.sh content here
"""

# Required files - job must contain exactly these files (include the script file)
REQUIRED_FILE_PATHS = ["code/main.py", "code/data.json", "run.sh", "config.yaml"]

REQUIRED_FILE_CONTENTS = {
    SCRIPT_PATH: EXPECTED_SCRIPT,
    "run.sh": EXPECTED_RUN_SCRIPT,
}

# Optional: list of allowed user emails (None = allow all)
ALLOWED_USERS = None

# Optional: only allow jobs from approved peers
PEERS_ONLY = False


def main():
    # Criteria that approve nothing would do so on every poll, so say it now
    # rather than once a job arrives.
    validate_criteria_paths(REQUIRED_FILE_CONTENTS, REQUIRED_FILE_PATHS)

    client = login_do(
        email=EMAIL,
        token_path=TOKEN_PATH,
    )

    while True:
        try:
            auto_approve_and_run_jobs(
                client,
                required_file_contents=REQUIRED_FILE_CONTENTS,
                required_file_paths=REQUIRED_FILE_PATHS,
                allowed_users=ALLOWED_USERS,
                peers_only=PEERS_ONLY,
                verbose=False,
            )
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as exc:
            # A poll can fail on a transient fault, so keep polling. The
            # criteria themselves were checked once, before the loop.
            print(f"poll failed: {exc}", file=sys.stderr)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
