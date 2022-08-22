# stdlib
import os
from pathlib import Path
import shutil
from typing import Dict
from typing import Optional
from typing import Union

# third party
from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import Template
import requests
from tqdm import tqdm
import yaml

# relative
from .lib import manifest_template_path
from .lib import repo_src_path

HAGRID_TEMPLATE_PATH = str(manifest_template_path())


def read_yml_file(filename: str) -> Optional[Dict]:

    template = None

    with open(filename) as fp:
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


def setup_from_manifest_template(release_type: str) -> None:
    template = read_yml_file(HAGRID_TEMPLATE_PATH)

    if template is None:
        raise ValueError(
            f"Failed to read {HAGRID_TEMPLATE_PATH}. Please check the file name or path is correct."
        )

    git_hash = template["hash"]
    git_base_url = template["baseUrl"]
    target_dir = template["target_dir"]

    files_to_download = template["files"]

    for files in tqdm(files_to_download, desc="Copying files... "):
        for trg_file_path, src_file_path in files.items():
            if release_type == "development":
                copy_files_from_repo(
                    src_file_path=src_file_path, trg_file_path=trg_file_path
                )
            else:
                local_destination = get_local_abs_path(target_dir, trg_file_path)
                link_to_file = git_url_for_file(src_file_path, git_base_url, git_hash)
                download_file(
                    link_to_file=link_to_file, local_destination=local_destination
                )


def render_templates(env_vars: dict, release_type: str) -> None:
    template = read_yml_file(HAGRID_TEMPLATE_PATH)

    if template is None:
        raise ValueError("Failed to read hagrid template.")

    files_to_render = template["files"]
    target_dir = (
        repo_src_path() if release_type == "development" else template["target_dir"]
    )

    jinja_template = JinjaTemplate(target_dir)

    for files in tqdm(files_to_render, desc="Substituting vars... "):
        for trg_file_path, _ in files.items():
            jinja_template.substitute_vars(trg_file_path, env_vars)


class JinjaTemplate(object):
    def __init__(self, template_dir: Union[str, os.PathLike]) -> None:
        self.directory = os.path.expanduser(template_dir)
        self.environ = Environment(
            loader=FileSystemLoader(self.directory), autoescape=True
        )

    def read_template_from_path(self, filepath: str) -> Template:
        return self.environ.get_template(name=filepath)

    def substitute_vars(self, template_path: str, vars_to_substitute: dict) -> None:

        template = self.read_template_from_path(template_path)
        rendered_template = template.render(vars_to_substitute)
        self.save_to(rendered_template, template_path)

    def save_to(self, message: str, filename: str) -> None:
        base_dir = self.directory
        with open(os.path.join(base_dir, filename), "w") as fp:
            fp.write(message)


def download_file(link_to_file: str, local_destination: str) -> None:

    file_dir = os.path.dirname(local_destination)
    os.makedirs(file_dir, exist_ok=True)

    try:
        # download file
        response = requests.get(link_to_file)
        if response.status_code != 200:
            raise Exception(f"Failed to download: {link_to_file}")

        # Save file to the local destination
        open(local_destination, "wb").write(response.content)

    except Exception as e:
        raise e


def copy_files_from_repo(src_file_path: str, trg_file_path: str) -> None:

    repo_directory = Path(repo_src_path())

    local_src_path = repo_directory / src_file_path
    local_destination = repo_directory / trg_file_path

    file_dir = os.path.dirname(local_destination)
    os.makedirs(file_dir, exist_ok=True)
    try:
        shutil.copyfile(local_src_path, local_destination)
    except shutil.SameFileError:
        print(f"{src_file_path}: Source and destination represents the same file.")

    # If destination is a directory.
    except IsADirectoryError:
        print("Destination is a directory.")

    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

    except Exception as e:
        print(
            f"While copying file from src: {src_file_path} to dest: {trg_file_path}, following exception occurred: {e}"
        )
