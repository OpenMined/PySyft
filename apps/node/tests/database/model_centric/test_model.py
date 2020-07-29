import pytest
import sys

from . import BIG_INT
from .presets.model import MODEL_ATTRIBUTES

from random import randint
from src.app.main.sfl.models.ai_model import Model, ModelCheckPoint

sys.path.append(".")


@pytest.mark.parametrize("version", MODEL_ATTRIBUTES)
def test_create_model_object(version, database):
    new_model = Model(version=version)
    database.session.add(new_model)
    database.session.commit()
