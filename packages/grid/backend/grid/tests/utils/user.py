# stdlib
from typing import Optional

# grid absolute
from grid.api.users.models import UserCreate

# relative
from .common import random_email
from .common import random_lower_string


def create_user(
    email: Optional[str] = random_email(),
    password: Optional[str] = random_lower_string(),
    name: Optional[str] = random_lower_string(),
    institution: Optional[str] = random_lower_string(),
    role: Optional[str] = "Administrator",
    budget: Optional[float] = 0.0,
) -> UserCreate:
    user = {
        "email": email,
        "password": password,
        "name": name,
        "role": role,
        "budget": budget,
        "institution": institution,
    }
    return UserCreate(**user)
