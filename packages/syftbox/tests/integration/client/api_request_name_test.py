from pathlib import Path

import syftbox
from syftbox.client.plugins.apps import run_apps


class AppFixture:
    @staticmethod
    def create_app_with_name_check(apps_dir: Path, app_name: str) -> tuple[Path, str]:
        """Create a test app that outputs its api_request_name"""
        app_dir = apps_dir / app_name
        app_dir.mkdir(parents=True, exist_ok=True)

        # Create a Python script that uses api_request_name
        test_script = app_dir / "main.py"
        test_script.write_text("""
from syftbox.lib import Client

client = Client.load()
print(f"API_NAME:{client.api_request_name}")
""")

        # Create runner script
        SYFTBOX_SOURCE_PATH = Path(syftbox.__file__).parent.parent
        run_script = app_dir / "run.sh"
        run_script.write_text(f"""#!/bin/bash
set -e

uv venv && . .venv/bin/activate
uv pip install --upgrade --editable {SYFTBOX_SOURCE_PATH}
python3 main.py | tee app.log
deactivate
""")
        run_script.chmod(0o755)
        return app_dir


def test_api_request_name_in_app(tmp_path, mock_config):
    """Test that api_request_name returns correct name when called from within an app"""
    # Setup
    app_name = "test_app_that_echoes_name"
    app_dir = AppFixture.create_app_with_name_check(tmp_path, app_name)

    # Run the app
    run_apps(tmp_path, mock_config.path)

    # Verify the output
    app_log = app_dir / "app.log"
    assert app_log.exists()
    output = app_log.read_text().strip()
    assert f"API_NAME:{app_name}" in output
