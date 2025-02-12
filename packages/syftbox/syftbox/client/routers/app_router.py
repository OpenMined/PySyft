import json
import os
import subprocess
from pathlib import Path

import yaml
from aiofiles import open as aopen
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing_extensions import List

from syftbox.client.routers.common import APIContext
from syftbox.lib.types import PathLike

router = APIRouter()


def parse_frontmatter(file_path: Path) -> dict:
    """
    Parses frontmatter YAML from a README.md file and returns it as a Python dictionary.

    Args:
        file_path (str): Path to the README.md file.

    Returns:
        dict: The parsed YAML frontmatter as a dictionary. If no frontmatter is found, returns an empty dict.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Check for YAML frontmatter boundaries
    if lines[0].strip() == "---":
        yaml_lines = []
        for line in lines[1:]:
            if line.strip() == "---":
                break
            yaml_lines.append(line)

        # Parse the YAML content
        frontmatter = yaml.safe_load("".join(yaml_lines))
        return frontmatter if frontmatter else {}
    else:
        return {}


class AppDetails(BaseModel):
    name: str
    version: str
    source: str
    home: str
    icon: str
    path: str


def get_all_apps(apps_dir: PathLike) -> List[AppDetails]:
    """
    Get all apps in the given directory.

    Args:
        apps_dir (str): Path to the directory containing the apps.

    Returns:
        list: A list of AppDetails objects.
    """
    apps = []
    for app_dir in Path(apps_dir).iterdir():
        if app_dir.is_dir():
            readme_path = app_dir / "README.md"
            if readme_path.exists():
                frontmatter = parse_frontmatter(readme_path)
                app = AppDetails(
                    name=frontmatter.get("name", app_dir.name),
                    version=frontmatter.get("version", "0.0.1"),
                    source=frontmatter.get("source", ""),
                    home=frontmatter.get("home", ""),
                    icon=frontmatter.get("icon", ""),
                    path=str(app_dir),
                )
                apps.append(app)

    return apps


@router.get("/")
async def index(ctx: APIContext) -> JSONResponse:
    apps_dir = ctx.workspace.apps
    apps = get_all_apps(apps_dir)

    return JSONResponse(content=[app.model_dump() for app in apps])


@router.get("/status/{app_name}")
async def app_details(ctx: APIContext, app_name: str) -> JSONResponse:
    apps_dir = ctx.workspace.apps
    apps = get_all_apps(apps_dir)
    for app in apps:
        if app_name == app.name:
            return JSONResponse(content=app.model_dump())
    return JSONResponse(status_code=404, content={"message": "App not found"})


class InstallRequest(BaseModel):
    source: str
    version: str


@router.post("/install")
async def install_app(request: InstallRequest) -> JSONResponse:
    command = ["syftbox", "app", "install", request.source, "--called-by", "api"]
    try:
        # Run the command and capture output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stderr, result.stdout)
        # If successful, return JSON payload indicating success
        return {
            "status": "success",
            "message": f"App {request.source} version {request.version} installed successfully.",
            "output": result.stdout,
        }

    except subprocess.CalledProcessError as e:
        # Handle command failure, return JSON with error details
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Failed to install app {request.source}.", "output": e.stderr},
        )


@router.post("/command/{app_name}")
async def app_command(ctx: APIContext, app_name: str, request: dict) -> JSONResponse:
    apps_dir = ctx.workspace.apps
    apps = get_all_apps(apps_dir)

    for app in apps:
        if app_name == app.name:
            # Convert request dictionary to JSON string and wrap with single quotes for shell
            request_json = json.dumps(request)
            json_arg = f"--input='{request_json}'"  # Wrap entire JSON argument in single quotes
            command = f"uv run {app.path}/command.py {json_arg}"  # Complete command as a single string
            print("command", command)

            # Create env dict with explicit string types
            env: dict[str, str] = {
                **{k: str(v) for k, v in os.environ.items()},
                "SYFTBOX_CLIENT_CONFIG_PATH": str(ctx.config.path),
            }

            try:
                # Execute the command with the specified environment
                result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True, env=env)

                # Trim the output and attempt to parse it as JSON
                trimmed_output = result.stdout.strip()
                try:
                    json_output = json.loads(trimmed_output)
                    return JSONResponse(content=json_output)
                except json.JSONDecodeError:
                    # Return trimmed output as plain text if not valid JSON
                    return JSONResponse(content={"output": trimmed_output})
            except subprocess.CalledProcessError as e:
                print("error", e)
                return JSONResponse(status_code=500, content={"error": e.stderr.strip()})

    raise HTTPException(status_code=404, detail="App not found")


@router.get("/logs/{app_name}")
async def app_logs(
    ctx: APIContext,
    app_name: str,
    limit: int = 256,
    offset: int = 0,
) -> JSONResponse:
    apps_dir = ctx.workspace.apps
    app_dir = Path(apps_dir) / app_name
    if not app_dir.is_dir():
        raise HTTPException(status_code=404, detail="App not found")

    logs: List[str] = []
    log_file = app_dir / "logs" / f"{app_name}.log"
    try:
        if log_file.is_file():
            async with aopen(log_file, "r") as file:
                logs = await file.readlines()

        # Calculate pagination indices
        total_logs = len(logs)
        start_idx = max(0, total_logs - offset - limit)
        end_idx = total_logs - offset if offset > 0 else total_logs
        logs = logs[start_idx:end_idx]

        return JSONResponse(
            content={
                "logs": logs,
                "total": total_logs,
                "source": str(log_file),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")
