# stdlib
import os
from typing import Dict
from typing import Optional
from urllib.request import urlretrieve

# third party
# from jinja2 import Template
import yaml

HAGRID_TEMPLATE = "manifest_template.yml"


def read_yml_file(filename: str) -> Optional[Dict]:

    # stdlib
    from importlib import resources as pkg_resources

    template = None

    with pkg_resources.open_text("hagrid", filename) as fp:
        try:
            template = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            raise exc

    return template


def git_url_for_file(file_path: str, base_url: str, hash: str) -> str:
    return os.path.join(base_url, hash, file_path)


def get_local_abs_path(target_dir: str, file_path: str) -> str:
    local_path = os.path.join(target_dir, file_path)
    return os.path.expanduser(local_path)


def setup_from_manifest() -> None:
    template = read_yml_file(HAGRID_TEMPLATE)

    if template is None:
        raise ValueError(
            f"Failed to read {HAGRID_TEMPLATE}. Please check the file name or path is correct."
        )

    git_hash = template["hash"]
    git_base_url = template["baseUrl"]
    target_dir = template["target_dir"]

    files_to_download = template["files"]

    for files in files_to_download:
        for trg_file_path, src_file_path in files.items():
            print(src_file_path, trg_file_path)
            link_to_file = git_url_for_file(src_file_path, git_base_url, git_hash)
            local_destination = get_local_abs_path(target_dir, trg_file_path)
            download_file(
                link_to_file=link_to_file, local_destination=local_destination
            )
            # TODO: Add step for updating vars using Jinja Templating.


def download_file(link_to_file: str, local_destination: str) -> str:

    file_dir = os.path.dirname(local_destination)
    print("File Dir: ", file_dir)
    os.makedirs(file_dir, exist_ok=True)

    try:
        print("Link to file", link_to_file)
        resultFilePath, _ = urlretrieve(link_to_file, local_destination)
    except Exception as e:
        raise e
    return resultFilePath
