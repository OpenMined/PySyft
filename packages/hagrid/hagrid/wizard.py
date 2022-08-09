# stdlib
from typing import Dict

# relative
from .deps import check_grid_docker
from .deps import check_hagrid
from .deps import check_syft
from .deps import check_syft_deps
from .nb_output import NBOutput


class Wizard:
    @property
    def check_hagrid(self) -> NBOutput:
        return check_hagrid()

    @property
    def check_syft_deps(self) -> NBOutput:
        return check_syft_deps()

    @property
    def check_syft(self) -> NBOutput:
        return check_syft()

    @property
    def check_syft_pre(self) -> NBOutput:
        return check_syft(pre=True)

    @property
    def check_grid_docker(self) -> NBOutput:
        return check_grid_docker()

    # def download(self, tutorial_name: str, reset: bool = False) -> NBOutput:
    #     if tutorial_name not in TUTORIALS.keys():
    #         return NBOutput(
    #             f'<div class="alert alert-danger">{tutorial_name} is not a valid tutorial name.</div>'
    #         )
    #     else:
    #         tutorial = TUTORIALS[tutorial_name]
    #         file_path, downloaded = quickstart_download_notebook(
    #             tutorial.url, directory, reset=reset
    #         )
    #         jupyter_path = file_path.replace(os.path.abspath(directory) + "/", "")
    #         html = ""
    #         if downloaded:
    #             html += f'<div class="alert alert-success">{tutorial_name} downloaded.'
    #         else:
    #             html += f'<div class="alert alert-info">{tutorial_name} already exists.'
    #         html += f'<br />📖 <a href="{jupyter_path}">Click to Open Tutorial</a></div>'
    #         return NBOutput(html)

    # def _repr_html_(self) -> str:
    #     html = ""
    #     html += "<h2>Tutorials</h2>"
    #     html += "<ul>"
    #     for name, tutorial in TUTORIALS.items():
    #         html += f"<li style='list-style:none;'>📖 <strong>{name}</strong><br />{tutorial.description}</li>"
    #     html += "</ul>"
    #     first = list(TUTORIALS.keys())[0]
    #     html += f'<blockquote>Try running: <br /><code>quickstart.download("{first}")</code></blockquote>'
    #     return html


wizard = Wizard()
