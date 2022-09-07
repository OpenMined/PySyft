# stdlib
import os
from typing import Dict
from typing import List
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
from .cache import RENDERED_DIR
from .lib import manifest_template_path
from .lib import repo_src_path
from .mode import EDITABLE_MODE

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


def setup_from_manifest_template(host_type: str) -> Dict:
    template = read_yml_file(HAGRID_TEMPLATE_PATH)
    kwargs_to_parse = {}

    if template is None:
        raise ValueError(
            f"Failed to read {HAGRID_TEMPLATE_PATH}. Please check the file name or path is correct."
        )

    git_hash = template["hash"]
    git_base_url = template["baseUrl"]
    target_dir = template["target_dir"]
    all_template_files = template["files"]
    docker_tag = template["dockerTag"]
    files_to_download = []

    for package_name in all_template_files:

        # Get all files w.r.t that package e.g. grid, syft, hagrid
        template_files = all_template_files[package_name]
        package_path = template_files["path"]
        to_absolute_file_path = lambda x: os.path.join(package_path, x)  # noqa: E731

        # common files
        files_to_download += list(map(to_absolute_file_path, template_files["common"]))

        # docker related files
        if host_type in ["docker"]:
            files_to_download += list(
                map(to_absolute_file_path, template_files["docker"])
            )

        # add k8s related files
        # elif host_type in ["k8s"]:
        #     files_to_download += template_files["k8s"]

        else:
            raise Exception(f"Hagrid template does not currently support {host_type}.")

    download_files(
        files_to_download=files_to_download,
        git_hash=git_hash,
        git_base_url=git_base_url,
        target_dir=target_dir,
    )

    kwargs_to_parse["tag"] = docker_tag

    return kwargs_to_parse


def download_files(
    files_to_download: List[str],
    git_hash: str,
    git_base_url: str,
    target_dir: str,
) -> None:

    if EDITABLE_MODE:
        print("Skipping copying files when running in editable mode.")
        return

    for src_file_path in tqdm(files_to_download, desc="Copying files... "):

        # For now target file path is same as source file path
        trg_file_path = src_file_path
        local_destination = get_local_abs_path(target_dir, trg_file_path)
        link_to_file = git_url_for_file(src_file_path, git_base_url, git_hash)
        download_file(link_to_file=link_to_file, local_destination=local_destination)


def render_templates(env_vars: dict, host_type: str) -> None:
    template = read_yml_file(HAGRID_TEMPLATE_PATH)

    if template is None:
        raise ValueError("Failed to read hagrid template.")

    target_dir = repo_src_path() if EDITABLE_MODE else template["target_dir"]
    all_template_files = template["files"]

    jinja_template = JinjaTemplate(target_dir)

    files_to_render = []
    for package_name in all_template_files:
        template_files = all_template_files[package_name]
        package_path = template_files["path"]

        # Aggregate all the files to be rendered

        # common files
        files_to_render += template_files["common"]

        # docker related files
        if host_type in ["docker"]:
            files_to_render += template_files["docker"]

        # Render the files
        for file_path in files_to_render:
            src_file_path = os.path.join(package_path, file_path)
            target_file_path = (
                os.path.join(package_path, RENDERED_DIR, file_path)
                if EDITABLE_MODE
                else src_file_path
            )
            jinja_template.substitute_vars(src_file_path, env_vars, target_file_path)


class JinjaTemplate(object):
    def __init__(self, template_dir: Union[str, os.PathLike]) -> None:
        self.directory = os.path.expanduser(template_dir)
        self.environ = Environment(
            loader=FileSystemLoader(self.directory), autoescape=True
        )

    def read_template_from_path(self, filepath: str) -> Template:
        return self.environ.get_template(name=filepath)

    def substitute_vars(
        self, template_path: str, vars_to_substitute: dict, target_path: str
    ) -> None:

        template = self.read_template_from_path(template_path)
        rendered_template = template.render(vars_to_substitute)
        self.save_to(rendered_template, target_path)

    def save_to(self, message: str, filename: str) -> None:
        base_dir = self.directory
        filepath = os.path.abspath(os.path.join(base_dir, filename))

        # Create sub directories if does not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save template to filepath
        with open(filepath, "w") as fp:
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
