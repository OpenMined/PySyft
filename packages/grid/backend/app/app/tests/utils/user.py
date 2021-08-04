# stdlib
from typing import Optional

# grid absolute
from app.users.models import UserCreate

# relative
from .common import random_email
from .common import random_lower_string


def create_user(
    email: Optional[str] = random_email(),
    password: Optional[str] = random_lower_string(),
    name: Optional[str] = random_lower_string(),
    role: Optional[str] = "Administrator",
) -> UserCreate:
    user = {"email": email, "password": password, "name": name, "role": role}
    return UserCreate(**user)
