# stdlib
from dataclasses import dataclass
import os
from typing import Dict
from typing import Tuple

# third party
import click
import requests

# relative
from .nb_output import NBOutput

directory = os.path.expanduser("~/.hagrid/quickstart/")


def quickstart_download_notebook(
    url: str, directory: str, reset: bool = False, overwrite_all_notebooks: bool = False
) -> Tuple[str, bool]:
    os.makedirs(directory, exist_ok=True)
    file_name = os.path.basename(url).replace("%20", "_")
    file_path = os.path.abspath(directory + file_name)

    file_exists = os.path.isfile(file_path)
    if overwrite_all_notebooks:
        reset = True

    if file_exists and not reset:
        reset = click.confirm(
            f"You already have the notebook {file_name}. "
            "Are you sure you want to overwrite it?"
        )

    downloaded = False
    if not file_exists or file_exists and reset:
        print(f"Downloading the notebook: {file_name}")
        r = requests.get(url, allow_redirects=True)
        with open(os.path.expanduser(file_path), "wb") as f:
            f.write(r.content)
        downloaded = True
    return file_path, downloaded


@dataclass
class Tutorial:
    filename: str
    description: str
    url: str


REPO_RAW_PATH = "https://raw.githubusercontent.com/OpenMined/PySyft"

TUTORIALS = {
    "01-data-owner": Tutorial(
        filename="01-data-owner.ipynb",
        description="Setting up a Domain in Docker on your local machine",
        url=(
            f"{REPO_RAW_PATH}/1d9be47cbd181b10a8410e9b96567747780c45a7/notebooks"
            + "/quickstart/Tutorial_Notebooks/Tutorial_01_DataOwner.ipynb"
        ),
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
            file_path, downloaded = quickstart_download_notebook(
                tutorial.url, directory, reset=reset
            )
            jupyter_path = file_path.replace(os.path.abspath(directory) + "/", "")
            html = ""
            if downloaded:
                html += f'<div class="alert alert-success">{tutorial_name} downloaded.'
            else:
                html += f'<div class="alert alert-info">{tutorial_name} already exists.'
            html += f'<br />📖 <a href="{jupyter_path}">Click to Open Tutorial</a></div>'
            return NBOutput(html)

    def _repr_html_(self) -> str:
        html = ""
        html += "<h2>Tutorials</h2>"
        html += "<blockquote>🙏 Coming soon. We have already opened PRs with the first tutorials.</blockquote>"
        # html += "<ul>"
        # for name, tutorial in TUTORIALS.items():
        #     html += f"<li style='list-style:none;'>📖 <strong>{name}</strong><br />{tutorial.description}</li>"
        # html += "</ul>"
        # first = list(TUTORIALS.keys())[0]
        # html += f'<blockquote>Try running: <br /><code>quickstart.download("{first}")</code></blockquote>'
        return html
