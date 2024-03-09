# stdlib
import argparse
from collections import defaultdict
import subprocess


def run_lsof():
    """Run the lsof command and return its output."""
    try:
        process = subprocess.Popen(["lsof"], stdout=subprocess.PIPE, text=True)
        output, _ = process.communicate()
        return output
    except Exception as e:
        print(f"Error running lsof: {e}")
        return ""


def run_lsof_for_pid(pid):
    """Run the lsof command for a specific PID and return its output."""
    try:
        process = subprocess.Popen(
            ["lsof", "-p", str(pid)], stdout=subprocess.PIPE, text=True
        )
        output, _ = process.communicate()
        return output
    except Exception as e:
        print(f"Error running lsof for PID {pid}: {e}")
        return ""


def parse_lsof_output(lsof_output, verbose):
    """Parse the lsof output."""
    data = defaultdict(list)
    lines = lsof_output.splitlines()

    for line in lines[1:]:  # Skip header line
        parts = line.split(maxsplit=8)
        if len(parts) < 9 or "python" not in parts[0].lower():
            continue  # Skip lines that are not Python processes

        proc_name, pid, owner, fd_type, fd_info, _, _, _, file_path = parts
        # Skip site-packages paths if not in verbose mode
        filters = [
            "site-packages",
            "lib-dynload",
            "cellar",
            ".pyenv",
            "ttys",
            "/dev/null",
            "/dev/random",
            "/dev/urandom",
            "localhost",
        ]
        skip = False
        if not verbose:
            for filter in filters:
                if filter in file_path.lower():
                    skip = True
                    break
        if skip:
            continue

        data[pid].append(
            {
                "Owner": owner,
                "FD Type": fd_type,
                "FD Info": fd_info,
                "File Path": file_path,
            }
        )

    return data


def main(pid=None, verbose=False):
    lsof_output = run_lsof_for_pid(pid) if pid else run_lsof()
    files_by_pid = parse_lsof_output(lsof_output, verbose)

    for pid, files in files_by_pid.items():
        print(f"PID {pid} open files:")
        for file in files:
            print(f"  {file['File Path']} ({file['FD Type']} - {file['FD Info']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List open files for Python processes."
    )
    parser.add_argument(
        "pid",
        nargs="?",
        type=int,
        default=None,
        help="The PID of the Python process (optional).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include all file descriptors, including those in site-packages.",
    )
    args = parser.parse_args()

    main(args.pid, args.verbose)
