import pytest
from src.app.main.users import User

from .presets.user import user_metrics


@pytest.mark.parametrize(
    ("email", "hashed_password", "salt", "private_key", "role"), user_metrics,
)
def test_create_user_object(
    email, hashed_password, salt, private_key, role, database,
):
    new_user = User(
        email=email,
        hashed_password=hashed_password,
        salt=salt,
        private_key=private_key,
        role=role,
    )
    database.session.add(new_user)
    database.session.commit()
