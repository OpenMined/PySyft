from pydantic import EmailStr
from app.users.models import UserCreate
from typing import Optional
from .common import random_email, random_lower_string


def create_user(email: Optional[EmailStr] = random_email(), password: Optional[str] = random_lower_string(), name: Optional[str] = random_lower_string(), role: Optional[str] = "Administrator") -> UserCreate:
    return UserCreate(email, password, name, role)

