import subprocess
from pathlib import Path

from typing_extensions import Optional

from syftbox.lib.types import PathLike

ASSETS_FOLDER = Path(__file__).parents[2] / "assets"
ICONS_PKG = ASSETS_FOLDER / "icon.zip"


# Function to search for Icon\r file
def search_icon_file(src_path: Path) -> Optional[Path]:
    if not src_path.exists():
        return None
    for file_path in src_path.iterdir():
        if "Icon" in file_path.name and "\r" in file_path.name:
            return file_path
    return None


# if you knew the pain of this function
def find_icon_file(src_path: Path) -> Path:
    # First attempt to find the Icon\r file
    icon_file = search_icon_file(src_path)
    if icon_file:
        return icon_file

    if not ICONS_PKG.exists():
        # If still not found, raise an error
        raise FileNotFoundError(f"{ICONS_PKG} not found")

    try:
        # cant use other zip tools as they don't unpack it correctly
        subprocess.run(
            ["ditto", "-xk", str(ICONS_PKG), str(src_path.parent)],
            check=True,
        )

        # Try to find the Icon\r file again after extraction
        icon_file = search_icon_file(src_path)
        if icon_file:
            return icon_file

        raise FileNotFoundError(f"Icon file not found for {src_path}")
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to unzip icon.zip using macOS CLI tool.")


def copy_icon_file(icon_folder: PathLike, dest_folder: PathLike) -> None:
    dest_path = Path(dest_folder)
    icon_path = Path(icon_folder)
    src_icon_path = find_icon_file(icon_path)
    if not dest_path.exists():
        raise FileNotFoundError(f"Destination folder '{dest_folder}' does not exist.")

    # shutil wont work with these special icon files
    subprocess.run(["cp", "-p", src_icon_path, dest_folder], check=True)
    subprocess.run(["SetFile", "-a", "C", dest_folder], check=True)
