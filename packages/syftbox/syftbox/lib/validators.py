import re
import shutil
import tempfile

from typing_extensions import Tuple

from syftbox.lib.types import PathLike, to_path

DIR_NOT_EMPTY = "Directory is not empty"


def is_valid_dir(path: PathLike, check_empty: bool = True, check_writable: bool = True) -> Tuple[bool, str]:
    try:
        if not path:
            return False, "Empty path"

        # Convert to Path object if string
        dir_path = to_path(path)

        # Must not be a reserved path
        if dir_path.is_reserved():
            return False, "Reserved path"

        if dir_path.exists():
            if not dir_path.is_dir():
                return False, "Path is not a directory"

            if check_empty and any(dir_path.iterdir()):
                return False, DIR_NOT_EMPTY
        elif check_writable:
            # Try to create a temporary file to test write permissions on parent
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                testfile = tempfile.TemporaryFile(dir=dir_path)
                testfile.close()
                shutil.rmtree(dir_path)
            except Exception as e:
                return False, str(e)

        # all checks passed
        return True, ""
    except Exception as e:
        return False, str(e)


def is_valid_email(email: str) -> bool:
    # Define a regex pattern for a valid email
    # from: https://stackoverflow.com/a/21608610
    email_regex = r"\w+([-+.']\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*"

    # Use the match method to check if the email fits the pattern
    if re.match(email_regex, email):
        return True
    return False
