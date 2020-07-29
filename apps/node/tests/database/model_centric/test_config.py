import sys
from random import randint

import pytest
from src.app.main.model_centric.processes.config import Config

from . import BIG_INT
from .presets.config import configs

sys.path.append(".")


@pytest.mark.parametrize("client_config, server_config", configs)
def test_create_config_object(client_config, server_config, database):
    my_server_config = Config(id=randint(0, BIG_INT), config=server_config)
    my_client_config = Config(id=randint(0, BIG_INT), config=client_config)
    database.session.add(my_server_config)
    database.session.add(my_client_config)
    database.session.commit()
