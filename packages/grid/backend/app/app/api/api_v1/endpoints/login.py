# stdlib
from datetime import timedelta
import json
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

# syft absolute
from syft import Domain
from syft import serialize

# grid absolute
from app import crud
from app import models
from app import schemas
from app.api import deps
from app.core import security
from app.core.config import settings
from app.core.security import get_password_hash
from app.utils import generate_password_reset_token
from app.utils import send_reset_password_email
from app.utils import verify_password_reset_token
from app.core.node import domain

router = APIRouter()


@router.post("/login/access-token", response_model=str)
def login_access_token(
    db: Session = Depends(deps.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """

    is_valid = domain.users.login(email=form_data.username, password=form_data.password)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    else:
        user  = domain.users.first(email=form_data.username)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return Response(
        json.dumps(
            {
                "access_token": security.create_access_token(user.id, expires_delta=access_token_expires),
                "token_type": "bearer",
                "metadata": serialize(domain.get_metadata_for_client()).SerializeToString().decode("ISO-8859-1"),
                "key": user.private_key
            }
        ),
        media_type="application/json"
    )


@router.post("/login/test-token", response_model=schemas.User)
def test_token(current_user: Any = Depends(deps.get_current_user)) -> Any:
    """
    Test access token
    """
    return current_user


@router.post("/password-recovery/{email}", response_model=schemas.Msg)
def recover_password(email: str, domain: Domain = Depends(deps.get_db)) -> Any:
    """
    Password Recovery
    """
    db = domain.db
    user = crud.user.get_by_email(db, email=email)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system.",
        )
    password_reset_token = generate_password_reset_token(email=email)
    send_reset_password_email(
        email_to=user.email, email=email, token=password_reset_token
    )
    return {"msg": "Password recovery email sent"}


@router.post("/reset-password/", response_model=schemas.Msg)
def reset_password(
    token: str = Body(...),
    new_password: str = Body(...),
    domain: Domain = Depends(deps.get_db),
) -> Any:
    """
    Reset password
    """
    db = domain.db
    email = verify_password_reset_token(token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid token")
    user = crud.user.get_by_email(db, email=email)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system.",
        )
    elif not crud.user.is_active(user):
        raise HTTPException(status_code=400, detail="Inactive user")
    hashed_password = get_password_hash(new_password)
    user.hashed_password = hashed_password
    db.add(user)
    db.commit()
    return {"msg": "Password updated successfully"}
