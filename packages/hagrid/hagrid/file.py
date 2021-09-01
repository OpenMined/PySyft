# stdlib
import os


def user_hagrid_profile() -> str:
    dir_path = os.path.expanduser("~/.hagrid")
    os.makedirs(dir_path, exist_ok=True)
    return os.path.abspath(dir_path)
