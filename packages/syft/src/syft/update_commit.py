# stdlib
import os
import subprocess  # nosec
import sys


def get_commit_hash() -> str:
    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        output = subprocess.check_output(
            "git rev-parse --short HEAD".split(" "),
            cwd=cwd,  # nosec
        )
        return output.strip().decode("ascii")
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit hash: {e}")
        sys.exit(1)


def update_commit_variable(file_path: str, commit_hash: str) -> None:
    """Replace the __commit__ variable with the actual commit hash."""
    try:
        with open(file_path) as file:
            lines = file.readlines()

        with open(file_path, "w") as file:
            updated = False
            for line in lines:
                if "__commit__ = " in line:
                    file.write(f'__commit__ = "{commit_hash}"\n')
                    updated = True
                else:
                    file.write(line)
            if not updated:
                print("No __commit__ variable found in the file.")
    except OSError as e:
        print(f"Error reading or writing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_commit.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    commit_hash = get_commit_hash()
    update_commit_variable(file_path, commit_hash)
