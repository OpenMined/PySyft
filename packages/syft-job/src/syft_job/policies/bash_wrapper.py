"""The run.sh that calls a Python policy."""

RUN_SCRIPT_NAME = "run.sh"
PYTHON_POLICY_NAME = "policy.py"

PYTHON_RUN_SCRIPT = f"""#!/bin/bash
exec python3 "$(dirname "$0")/{PYTHON_POLICY_NAME}"
"""


def python_run_script() -> str:
    """run.sh for a policy folder holding policy.py."""
    return PYTHON_RUN_SCRIPT
