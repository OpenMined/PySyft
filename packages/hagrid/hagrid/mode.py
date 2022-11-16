# stdlib
import os
from pathlib import Path
import site


def hagrid_root() -> str:
    return os.path.abspath(str(Path(__file__).parent.parent))


def is_editable_mode() -> bool:
    current_package_root = hagrid_root()

    installed_as_editable = False
    sitepackages_dirs = site.getsitepackages()
    # check all site-packages returned if they have a hagrid.egg-link
    for sitepackages_dir in sitepackages_dirs:
        egg_link_file = Path(sitepackages_dir) / "hagrid.egg-link"
        try:
            linked_folder = egg_link_file.read_text()
            # if the current code is in the same path as the egg-link its -e mode
            installed_as_editable = current_package_root in linked_folder
            break
        except Exception:  # nosec
            pass

    if os.path.exists(Path(current_package_root) / "hagrid.egg-info"):
        installed_as_editable = True

    return installed_as_editable


EDITABLE_MODE = is_editable_mode()
