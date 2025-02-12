import json

import pytest

from syftbox.app.install import load_config


def test_load_app_config(tmp_path):
    valid_json_config = {
        "version": "0.1.0",
        "app": {
            "version": "0.1.0",
            "run": {"command": ["python", "main.py"], "interval": "10"},
            "env": {},
            "platforms": ["linux"],
            "pre_install": ["pip", "install", "psutil"],
            "post_install": [],
            "pre_update": [],
            "post_update": [],
        },
    }

    app_conf = tmp_path / "app.json"
    app_conf.write_text(json.dumps(valid_json_config))

    app_config = load_config(app_conf)
    assert app_config.version == valid_json_config["version"]
    assert app_config.app.version == valid_json_config["app"]["version"]
    assert app_config.app.run.command == valid_json_config["app"]["run"]["command"]
    assert vars(app_config.app.env) == valid_json_config["app"]["env"]
    assert app_config.app.platforms == valid_json_config["app"]["platforms"]
    assert app_config.app.pre_install == valid_json_config["app"]["pre_install"]
    assert app_config.app.pre_update == valid_json_config["app"]["pre_update"]
    assert app_config.app.post_update == valid_json_config["app"]["post_update"]


def test_load_invalid_app_config(tmp_path):
    app_conf = tmp_path / "app.json"
    app_conf.write_text("Invalid JSON")

    with pytest.raises(ValueError) as expt:
        load_config(app_conf)
        assert expt.value == "File isn't in JSON format"


def test_load_inexistent_app_config():
    with pytest.raises(ValueError) as expt:
        load_config("inexistent_app.json")
        assert expt.value == "Couln't find the json config file for this path."
