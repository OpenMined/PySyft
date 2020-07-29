import sys
from random import randint

import pytest
from src.app.main.model_centric.models.ai_model import Model, ModelCheckPoint

from . import BIG_INT
from .presets.model import MODEL_ATTRIBUTES

sys.path.append(".")


@pytest.mark.parametrize("version", MODEL_ATTRIBUTES)
def test_create_model_object(version, database):
    new_model = Model(version=version)
    database.session.add(new_model)
    database.session.commit()
