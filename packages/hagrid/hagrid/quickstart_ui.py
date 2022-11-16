# stdlib
from dataclasses import dataclass
import os
import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from urllib.parse import urlparse

# third party
import click
import requests
from tqdm import tqdm

# relative
from .cache import DEFAULT_BRANCH
from .cache import DEFAULT_REPO
from .cache import arg_cache
from .nb_output import NBOutput

directory = os.path.expanduser("~/.hagrid/quickstart/")


def quickstart_download_notebook(
    url: str, directory: str, reset: bool = False, overwrite_all: bool = False
) -> Tuple[str, bool, bool]:
    os.makedirs(directory, exist_ok=True)
    file_name = os.path.basename(url).replace("%20", "_").replace(" ", "_")
    file_path = os.path.abspath(directory + file_name)

    file_exists = os.path.isfile(file_path)
    if overwrite_all:
        reset = True

    if file_exists and not reset:
        response = click.prompt(
            f"\nOverwrite {file_name}?",
            prompt_suffix="(a/y/N)",
            default="n",
            show_default=False,
        )
        if response.lower() == "a":
            reset = True
            overwrite_all = True
        elif response.lower() == "y":
            reset = True
        else:
            print(f"Skipping {file_name}")
            reset = False

    downloaded = False
    if not file_exists or file_exists and reset:
        print(f"Downloading notebook: {file_name}")
        r = requests.get(url, allow_redirects=True)
        with open(os.path.expanduser(file_path), "wb") as f:
            f.write(r.content)
        downloaded = True
    return file_path, downloaded, overwrite_all


def fetch_notebooks_for_url(
    url: str,
    directory: str,
    reset: bool = False,
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    commit: Optional[str] = None,
) -> List[str]:
    downloaded_files = []
    allowed_schemes_as_url = ["http", "https"]
    url_scheme = urlparse(url).scheme
    # relative mode
    if url_scheme not in allowed_schemes_as_url:
        notebooks = get_urls_from_dir(repo=repo, branch=branch, commit=commit, url=url)

        url_dir = os.path.dirname(url) if os.path.dirname(url) else url
        notebook_files = []
        existing_count = 0
        for notebook_url in notebooks:
            url_filename = os.path.basename(notebook_url)
            url_dirname = os.path.dirname(notebook_url)
            if (
                url_dirname.endswith(url_dir)
                and os.path.isdir(directory + url_dir)
                and os.path.isfile(directory + url_dir + os.sep + url_filename)
            ):
                notebook_files.append(url_dir + os.sep + url_filename)
                existing_count += 1

        if existing_count > 0:
            plural = "s" if existing_count > 1 else ""
            print(
                f"You have {existing_count} existing notebook{plural} matching: {url}"
            )
            for nb in notebook_files:
                print(nb)

        overwrite_all = False
        for notebook_url in tqdm(notebooks):
            file_path, _, overwrite_all = quickstart_download_notebook(
                url=notebook_url,
                directory=directory + os.sep + url_dir + os.sep,
                reset=reset,
                overwrite_all=overwrite_all,
            )
            downloaded_files.append(file_path)

    else:
        file_path, _, _ = quickstart_download_notebook(
            url=url, directory=directory, reset=reset
        )
        downloaded_files.append(file_path)
    return downloaded_files


@dataclass
class Tutorial:
    filename: str
    description: str
    url: str


REPO_RAW_PATH = "https://raw.githubusercontent.com/OpenMined/PySyft"

TUTORIALS = {
    "data-owner": Tutorial(
        filename="data-owner",
        description="Deploying a Test Domain and Uploading Data",
        url="data-owner",
    )
}


class QuickstartUI:
    @property
    def tutorials(self) -> Dict[str, Tutorial]:
        return TUTORIALS

    def download(self, tutorial_name: str, reset: bool = False) -> NBOutput:
        if tutorial_name not in TUTORIALS.keys():
            return NBOutput(
                f'<div class="alert alert-danger">{tutorial_name} is not a valid tutorial name.</div>'
            )
        else:
            tutorial = TUTORIALS[tutorial_name]

            downloaded_files = fetch_notebooks_for_url(
                url=tutorial.url, directory=directory
            )
            html = ""
            if len(downloaded_files) == 0:
                html += f'<div class="alert alert-danger">{tutorial_name} failed to download.'
            else:
                first = downloaded_files[0]
                jupyter_path = first.replace(os.path.abspath(directory) + "/", "")

                html += f'<div class="alert alert-success">{tutorial_name} downloaded.'
                html += (
                    f'<br />📖 <a href="{jupyter_path}">Click to Open Tutorial</a></div>'
                )
            return NBOutput(html)

    def _repr_html_(self) -> str:
        html = ""
        if not arg_cache.install_wizard_complete:
            html += "<h3>Step 1b: Install 🧙🏽‍♂️ Wizard (Recommended)</h3>"
            html += (
                "It looks like this might be your first time running Quickstart.<br />"
            )
            html += (
                "<blockquote>Please go through the Install Wizard notebook to "
                + "install Syft and optionally start a Grid server."
            )
            html += (
                '<br />📖 <a href="./01-install-wizard.ipynb">Click to start '
                + "Install 🧙🏽‍♂️ Wizard</a></div></blockquote>"
            )
            html += "<br />"

        html += "<h3>Download Tutorials</h3>"
        html += "Below is a list of tutorials to download using quickstart.<br />"
        html += "<ul>"
        for name, tutorial in TUTORIALS.items():
            html += (
                "<li style='list-style:none;'>📖 Tutorial Series: "
                + f"<strong>{name}</strong><br />{tutorial.description}</li>"
            )
        html += "</ul>"
        first = list(TUTORIALS.keys())[0]
        html += (
            "<blockquote>Try running: <br /><code>"
            + f'quickstart.download("{first}")</code></blockquote>'
        )

        return html


def get_urls_from_dir(
    url: str,
    repo: str,
    branch: str,
    commit: Optional[str] = None,
) -> List[str]:
    notebooks = []
    slug = commit if commit else branch

    gh_api_call = (
        "https://api.github.com/repos/" + repo + "/git/trees/" + slug + "?recursive=1"
    )
    r = requests.get(gh_api_call)
    if r.status_code != 200:
        print(
            f"Failed to fetch notebook from: {gh_api_call}.\n"
            + "Please try again with the correct parameters!"
        )
        sys.exit(1)

    res = r.json()

    for file in res["tree"]:
        if file["path"].startswith("notebooks/quickstart/" + url):
            if file["path"].endswith(".ipynb"):
                temp_url = (
                    "https://raw.githubusercontent.com/"
                    + repo
                    + "/"
                    + slug
                    + "/"
                    + file["path"]
                )
                notebooks.append(temp_url)

    if len(notebooks) == 0:
        for file in res["tree"]:
            if file["path"].startswith("notebooks/" + url):
                if file["path"].endswith(".ipynb"):
                    temp_url = (
                        "https://raw.githubusercontent.com/"
                        + repo
                        + "/"
                        + slug
                        + "/"
                        + file["path"]
                    )
                    notebooks.append(temp_url)
    return notebooks
