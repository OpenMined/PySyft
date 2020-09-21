import pytest

from src.app.database.user import User
from .presets.user import user_metrics


@pytest.mark.parametrize(
    ("email, hashed_password, salt, role"), user_metrics,
)
def test_create_user_object(email, hashed_password, salt, role, database):
    user = User(email=email, hashed_password=hashed_password, salt=salt, role=role)
    database.session.add(user)
    database.session.commit()
