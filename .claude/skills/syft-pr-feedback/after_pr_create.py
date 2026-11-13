"""Claude Code PostToolUse hook: after `gh pr create`, ask Claude to run syft-pr-feedback.

Enable it for yourself in .claude/settings.local.json; docs/engineering/README.md shows how.
Reads the hook input from stdin. Prints nothing unless the Bash call created a pull request.
"""

import json
import re
import sys

PR_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/(\d+)")


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    command = (event.get("tool_input") or {}).get("command", "")
    if event.get("tool_name") != "Bash" or "gh pr create" not in command:
        return
    response = event.get("tool_response") or {}
    output = response.get("stdout", "") if isinstance(response, dict) else str(response)
    match = PR_URL.search(output)
    if not match:
        return
    number = match.group(1)
    message = (
        f"Pull request #{number} was just created. Run the syft-pr-feedback skill on it "
        f"(/syft-pr-feedback {number}), then show the user the report and ask what to fix or post."
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": message,
                }
            }
        )
    )


if __name__ == "__main__":
    main()
