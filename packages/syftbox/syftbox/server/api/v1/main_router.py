import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from jinja2 import Template
from loguru import logger
from typing_extensions import Any, Union

from syftbox import __version__
from syftbox.lib.lib import (
    get_datasites,
)
from syftbox.server.analytics import log_analytics_event
from syftbox.server.settings import ServerSettings, get_server_settings
from syftbox.server.users.auth import get_current_user

current_dir = Path(__file__).parent

ascii_art = rf"""
 ____         __ _   ____
/ ___| _   _ / _| |_| __ )  _____  __
\___ \| | | | |_| __|  _ \ / _ \ \/ /
 ___) | |_| |  _| |_| |_) | (_) >  <
|____/ \__, |_|  \__|____/ \___/_/\_\
       |___/        {__version__:>17}


# Install Syftbox (MacOS and Linux)
curl -LsSf [[SERVER_URL]]/install.sh | sh

# Run the client
syftbox client
"""

main_router = APIRouter()


@main_router.get("/", response_class=PlainTextResponse)
async def get_ascii_art(request: Request) -> str:
    return ascii_art.replace("[[SERVER_URL]]", str(request.url).rstrip("/"))


def get_file_list(directory: Union[str, Path] = ".") -> list[dict[str, Any]]:
    # TODO rewrite with pathlib
    directory = str(directory)

    file_list = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        is_dir = os.path.isdir(item_path)
        size = os.path.getsize(item_path) if not is_dir else "-"
        mod_time = datetime.fromtimestamp(os.path.getmtime(item_path)).strftime("%Y-%m-%d %H:%M:%S")

        file_list.append({"name": item, "is_dir": is_dir, "size": size, "mod_time": mod_time})

    return sorted(file_list, key=lambda x: (not x["is_dir"], x["name"].lower()))


@main_router.get("/datasites", response_class=HTMLResponse)
async def list_datasites(
    request: Request, server_settings: ServerSettings = Depends(get_server_settings)
) -> HTMLResponse:
    files = get_file_list(server_settings.snapshot_folder)
    template_path = current_dir.parent.parent / "templates" / "datasites.html"
    html = ""
    with open(template_path) as f:
        html = f.read()
    template = Template(html)

    html_content = template.render(
        {
            "request": request,
            "files": files,
            "current_path": "/",
        }
    )
    return html_content


@main_router.get("/datasites/{path:path}", response_class=HTMLResponse)
async def browse_datasite(
    request: Request,
    path: str,
    server_settings: ServerSettings = Depends(get_server_settings),
) -> HTMLResponse:
    if path == "":  # Check if path is empty (meaning "/datasites/")
        return RedirectResponse(url="/datasites")

    snapshot_folder = str(server_settings.snapshot_folder)
    datasite_part = path.split("/")[0]
    datasites = get_datasites(snapshot_folder)
    if datasite_part in datasites:
        slug = path[len(datasite_part) :]
        if slug == "":
            slug = "/"
        datasite_path = os.path.join(snapshot_folder, datasite_part)
        datasite_public = datasite_path + "/public"
        if not os.path.exists(datasite_public):
            return "No public datasite"

        slug_path = os.path.abspath(datasite_public + slug)
        if os.path.exists(slug_path) and os.path.isfile(slug_path):
            if slug_path.endswith(".html") or slug_path.endswith(".htm"):
                return FileResponse(slug_path)
            elif slug_path.endswith(".md"):
                with open(slug_path, "r") as file:
                    content = file.read()
                return PlainTextResponse(content)
            elif slug_path.endswith(".json") or slug_path.endswith(".jsonl"):
                return FileResponse(slug_path, media_type="application/json")
            elif slug_path.endswith(".yaml") or slug_path.endswith(".yml"):
                return FileResponse(slug_path, media_type="application/x-yaml")
            elif slug_path.endswith(".log") or slug_path.endswith(".txt"):
                return FileResponse(slug_path, media_type="text/plain")
            elif slug_path.endswith(".py"):
                return FileResponse(slug_path, media_type="text/plain")
            else:
                return FileResponse(slug_path, media_type="application/octet-stream")

        # show directory
        if not path.endswith("/") and os.path.exists(path + "/") and os.path.isdir(path + "/"):
            return RedirectResponse(url=f"{path}/")

        index_file = os.path.abspath(slug_path + "/" + "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r") as file:
                html_content = file.read()
            return HTMLResponse(content=html_content, status_code=200)

        if os.path.isdir(slug_path):
            files = get_file_list(slug_path)
            template_path = current_dir.parent.parent / "templates" / "folder.html"
            html = ""
            with open(template_path) as f:
                html = f.read()
            template = Template(html)
            html_content = template.render(
                {
                    "datasite": datasite_part,
                    "request": request,
                    "files": files,
                    "current_path": path,
                }
            )
            return html_content
        else:
            # return 404
            message_404 = f"No file or directory found at /datasites/{datasite_part}{slug}"
            return HTMLResponse(content=message_404, status_code=404)

    return f"No Datasite {datasite_part} exists"


@main_router.post("/register")
async def register(
    request: Request,
    server_settings: ServerSettings = Depends(get_server_settings),
) -> JSONResponse:
    data = await request.json()
    email = data["email"]

    # create datasite snapshot folder
    datasite_folder = Path(server_settings.snapshot_folder) / email
    os.makedirs(datasite_folder, exist_ok=True)

    logger.info(f"> {email} registering, snapshot folder: {datasite_folder}")
    log_analytics_event("/register", email)

    return JSONResponse({"status": "success", "token": "0"}, status_code=200)


@main_router.post("/log_event")
async def log_event(
    request: Request,
    email: str = Depends(get_current_user),
) -> JSONResponse:
    data = await request.json()
    log_analytics_event("/log_event", email, **data)
    return JSONResponse({"status": "success"}, status_code=200)


@main_router.get("/install.sh")
async def install() -> FileResponse:
    install_script = current_dir.parent.parent / "templates" / "install.sh"
    return FileResponse(install_script, media_type="text/plain")


@main_router.get("/icon.png")
async def icon() -> FileResponse:
    icon_path = current_dir.parent.parent / "assets" / "icon.png"
    return FileResponse(icon_path, media_type="image/png")


@main_router.get("/info")
async def info() -> dict:
    return {
        "version": __version__,
    }
