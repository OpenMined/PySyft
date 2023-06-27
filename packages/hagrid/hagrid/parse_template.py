# stdlib
import hashlib
import os
import shutil
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.parse import urlparse

# third party
from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import Template
import requests
from rich.progress import track
import yaml

# relative
from .cache import DEFAULT_REPO
from .cache import STABLE_BRANCH
from .grammar import GrammarTerm
from .grammar import HostGrammarTerm
from .lib import hagrid_cache_dir
from .lib import manifest_template_path
from .lib import repo_src_path
from .mode import EDITABLE_MODE

HAGRID_TEMPLATE_PATH = str(manifest_template_path())


def read_yml_file(filename: str) -> Tuple[Optional[Dict], str]:
    template = None

    with open(filename) as fp:
        try:
            text = fp.read()
            template = yaml.safe_load(text)
            template_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        except yaml.YAMLError as exc:
            raise exc

    return template, template_hash


def read_yml_url(yml_url: str) -> Tuple[Optional[Dict], str]:
    template = None

    try:
        # download file
        response = requests.get(yml_url)  # nosec
        if response.status_code != 200:
            raise Exception(f"Failed to download: {yml_url}")

        # Save file to the local destination
        try:
            template = yaml.safe_load(response.content)
            template_hash = hashlib.sha256(response.content).hexdigest()
        except yaml.YAMLError as exc:
            raise exc

    except Exception as e:
        raise e

    return template, template_hash


def git_url_for_file(file_path: str, base_url: str, hash: str) -> str:
    # url must have unix style slashes
    return os.path.join(base_url, hash, file_path).replace(os.sep, "/")


def get_local_abs_path(target_dir: str, file_path: str) -> str:
    local_path = os.path.join(target_dir, file_path)
    return os.path.expanduser(local_path)


def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_path(string: str) -> bool:
    return os.path.exists(string)


def manifest_cache_path(template_hash: str) -> str:
    return f"{hagrid_cache_dir()}/manifests/{template_hash}"


def url_from_repo(template_location: Optional[str]) -> Optional[str]:
    if template_location is None:
        return None

    if ":" in template_location and "/" in template_location:
        parts = template_location.split(":")
        branch_or_hash = parts[1]
        repo = parts[0]
    elif ":" not in template_location and "/" in template_location:
        branch_or_hash = STABLE_BRANCH
        repo = template_location
    else:
        branch_or_hash = template_location
        repo = DEFAULT_REPO

    manifest_url = (
        f"https://raw.githubusercontent.com/{repo}/{branch_or_hash}"
        "/packages/hagrid/hagrid/manifest_template.yml"
    )

    if is_url(manifest_url):
        return manifest_url
    return None


def get_template_yml(template_location: Optional[str]) -> Tuple[Optional[Dict], str]:
    if template_location:
        if is_url(template_location):
            template, template_hash = read_yml_url(template_location)
        elif is_path(template_location):
            template, template_hash = read_yml_file(template_location)
        elif url_from_repo(template_location):
            template, template_hash = read_yml_url(url_from_repo(template_location))
        else:
            raise Exception(f"{template_location} is not valid")
    else:
        template_location = HAGRID_TEMPLATE_PATH

        template, template_hash = read_yml_file(template_location)

    if EDITABLE_MODE and is_path(template_location):
        # save it to the same folder for dev mode
        template_hash = "dev"
    return template, template_hash


def setup_from_manifest_template(
    host_type: str,
    node_type: Union[GrammarTerm, HostGrammarTerm],
    template_location: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> Dict:
    template, template_hash = get_template_yml(template_location)

    kwargs_to_parse = {}

    if template is None:
        raise ValueError(
            f"Failed to read {template_location}. Please check the file name or path is correct."
        )

    git_hash = template["hash"]
    git_base_url = template["baseUrl"]
    target_dir = manifest_cache_path(template_hash)
    all_template_files = template["files"]
    docker_tag = template["dockerTag"]
    files_to_download = []

    for package_name in all_template_files:
        # Get all files w.r.t that package e.g. grid, syft, hagrid
        template_files = all_template_files[package_name]
        package_path = template_files["path"]

        # common files
        files_to_download += [
            os.path.join(package_path, f) for f in template_files["common"]
        ]

        # enclave
        if node_type.input == "enclave" and host_type in ["docker"]:
            files_to_download += [
                os.path.join(package_path, f) for f in template_files["enclave"]
            ]
        # docker related files
        elif host_type in ["docker"]:
            files_to_download += [
                os.path.join(package_path, f) for f in template_files["docker"]
            ]

        # add k8s related files
        # elif host_type in ["k8s"]:
        #     files_to_download += template_files["k8s"]

        else:
            raise Exception(f"Hagrid template does not currently support {host_type}.")

    if EDITABLE_MODE and is_path(template_location):
        # to test things in editable mode we can pass in a .yml file path and it will
        # copy the files instead of download them
        for src_file_path in track(files_to_download, description="Copying files"):
            full_src_dir = f"{repo_src_path()}/{src_file_path}"
            full_target_path = f"{target_dir}/{src_file_path}"
            full_target_dir = os.path.dirname(full_target_path)
            os.makedirs(full_target_dir, exist_ok=True)

            shutil.copyfile(
                full_src_dir,
                full_target_path,
            )
    else:
        download_files(
            files_to_download=files_to_download,
            git_hash=git_hash,
            git_base_url=git_base_url,
            target_dir=target_dir,
            overwrite=overwrite,
            verbose=verbose,
        )

    kwargs_to_parse["tag"] = docker_tag
    return kwargs_to_parse


def deployment_dir(node_name: str) -> str:
    return f"{hagrid_cache_dir()}/deployments/{node_name}"


def download_files(
    files_to_download: List[str],
    git_hash: str,
    git_base_url: str,
    target_dir: str,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    for src_file_path in track(files_to_download, description="Downloading files"):
        # For now target file path is same as source file path
        trg_file_path = src_file_path
        local_destination = get_local_abs_path(target_dir, trg_file_path)
        link_to_file = git_url_for_file(src_file_path, git_base_url, git_hash)
        download_file(
            link_to_file=link_to_file,
            local_destination=local_destination,
            overwrite=overwrite,
            verbose=verbose,
        )


def render_templates(
    node_name: str,
    node_type: Union[GrammarTerm, HostGrammarTerm],
    template_location: Optional[str],
    env_vars: dict,
    host_type: str,
) -> None:
    template, template_hash = get_template_yml(template_location)

    if template is None:
        raise ValueError("Failed to read hagrid template.")

    src_dir = manifest_cache_path(template_hash)
    target_dir = deployment_dir(node_name)
    all_template_files = template["files"]

    jinja_template = JinjaTemplate(src_dir)

    files_to_render = []
    for package_name in all_template_files:
        template_files = all_template_files[package_name]

        # Aggregate all the files to be rendered

        # common files
        files_to_render += template_files["common"]

        # enclave
        if node_type.input == "enclave" and host_type in ["docker"]:
            for template_file in template_files["enclave"]:
                if "default.env" not in template_file:
                    files_to_render.append(template_file)

        elif host_type in ["docker"]:
            # docker related files
            for template_file in template_files["docker"]:
                if "default.env" not in template_file:
                    files_to_render.append(template_file)

        # Render the files
        for file_path in files_to_render:
            folder_path = template_files["path"]
            # relative to src_dir
            src_file_path = f"{folder_path}{file_path}"
            target_file_path = f"{target_dir}/{file_path}"
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
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


def download_file(
    link_to_file: str,
    local_destination: str,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    file_dir = os.path.dirname(local_destination)
    os.makedirs(file_dir, exist_ok=True)

    if not os.path.exists(local_destination) or overwrite:
        try:
            # download file
            response = requests.get(link_to_file)  # nosec
            if response.status_code != 200:
                raise Exception(f"Failed to download: {link_to_file}")

            # Save file to the local destination
            open(local_destination, "wb").write(response.content)

        except Exception as e:
            raise e
    else:
        if verbose:
            print(f"Skipping download: {link_to_file} exists.")
