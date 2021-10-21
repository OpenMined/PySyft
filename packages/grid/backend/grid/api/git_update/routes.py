# stdlib
from pathlib import Path
from typing import Any
from typing import Optional

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic.main import BaseModel

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.api.requests.routes import raise_generic_private_error
from grid.api.users.models import UserPrivate
from grid.core.config import settings

router = APIRouter()


class GitSettings(BaseModel):
    branch: str
    hash: str
    repository: str


class GitSettingsUpdate(GitSettings):
    branch: Optional[str]
    hash: Optional[str]
    repository: Optional[str]


def load_git_settings_from_file(filename):
    file = Path(filename)
    git_settings = GitSettings.parse_file(file)
    return git_settings


@router.patch("", status_code=200, response_class=JSONResponse)
def update_git_settings(
    # current_user: UserPrivate = Depends(get_current_user),
    git_settings: GitSettingsUpdate = Body(...),
) -> Any:
    # if current_user is None:
    ## TODO: Set to auth error
    # raise_generic_private_error()

    try:
        current_git_settings = load_git_settings_from_file(settings.GIT_SETTINGS_FILE)
        with open(Path(settings.GIT_SETTINGS_FILE), "w") as git_file:
            update = git_settings.dict(exclude_unset=True)
            updated_settings = current_git_settings.copy(update=update)
            git_file.write(updated_settings.json())
        return updated_settings
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()


@router.get("", status_code=200, response_class=JSONResponse)
def get_git_settings() -> Any:
    try:
        return load_git_settings_from_file(settings.GIT_SETTINGS_FILE)
    except Exception as err:
        logger.error(err)
        raise_generic_private_error()
