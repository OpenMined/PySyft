import os
import subprocess
from typing import Tuple

from syftbox.lib.platform import OS_IS_WSL2, UNAME


def open_dir(folder_path: str) -> Tuple[bool, str]:
    """
    Open the specified folder in the default file explorer.

    Args:
        folder_path (str): The path to the folder to be opened.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating success or failure,
                          and a message describing the result.

    Description:
        This function attempts to open the folder specified by `folder_path` in the
        default file explorer of the operating system. It handles different platforms
        (Windows, macOS, Linux) and uses platform-specific commands to open the folder.
        - On Windows, it uses `explorer`.
        - On macOS, it uses `open`.
        - On Linux, it uses `xdg-open` or `explorer.exe` in WSL environments.

        If the folder does not exist or an error occurs while attempting to open it, the
        function returns `False` along with an appropriate error message.
    """
    folder_path = os.path.expanduser(folder_path)
    if not os.path.exists(folder_path):
        return False, f"Folder does not exist: {folder_path}"

    try:
        system_name = UNAME.system
        if system_name == "Darwin":
            subprocess.run(["open", folder_path])
        elif system_name == "Windows":
            subprocess.run(["explorer", folder_path])
        elif system_name == "Linux":
            if OS_IS_WSL2:
                # Convert the path to Windows format for explorer.exe
                windows_path = _convert_to_windows_path(folder_path)
                if windows_path:
                    subprocess.run(["explorer.exe", windows_path])
                else:
                    return False, "Failed to convert path to Windows format for WSL"
            else:
                # Use the default Linux file explorer
                distro_explorer = _get_linux_file_explorer()
                subprocess.run([distro_explorer, folder_path])
        else:
            return False, f"Unsupported OS for opening folders: {system_name}"
        return True, "Folder opened successfully"
    except Exception as e:
        return False, str(e)


def _convert_to_windows_path(folder_path: str) -> str:
    """
    Convert a Linux path to a Windows path in WSL.

    Args:
        folder_path (str): The Linux path to be converted.

    Returns:
        str: The corresponding Windows path, or an empty string if the conversion fails.

    Description:
        This function uses the `wslpath` command to convert a Linux path to a Windows
        path when running in a WSL environment. If an error occurs during the
        conversion, it returns an empty string.
    """
    try:
        # Use wslpath to convert the path
        result = subprocess.run(["wslpath", "-w", folder_path], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def _get_linux_file_explorer() -> str:
    """
    Get the default file explorer for Linux distributions.

    Returns:
        str: The command used to open folders in the default Linux file explorer.

    Description:
        This function returns the command used to open folders in Linux. By default,
        it returns "xdg-open", which is commonly available on many Linux
        distributions to open files and folders with the default application.
    """
    # implement as needed, for now just return xdg-open
    return "xdg-open"
