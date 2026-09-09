"""Shared helpers for the syft-rds product tests.

Uniquely named (not ``utils.py``) to avoid a pytest module-name collision with
``tests/unit/utils.py`` when the full monorepo suite is collected together.
"""

import random
from pathlib import Path


def create_tmp_dataset_files():
    tmp_dir = Path("/tmp/syft-datasets-testing") / str(random.randint(1, 1000000))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mock_path = tmp_dir / "mock.txt"
    private_path = tmp_dir / "private.txt"
    readme_path = tmp_dir / "readme.md"
    mock_path.write_text("Hello, world!")
    private_path.write_text("Hello, world private!")
    readme_path.write_text("Hello, world!")
    return mock_path, private_path, readme_path


def create_tmp_dataset_files_with_parquet():
    """Create temporary dataset files with parquet files (binary format)."""
    import pandas as pd

    tmp_dir = Path("/tmp/syft-datasets-testing") / str(random.randint(1, 1000000))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Create parquet files (binary format)
    mock_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "score": [85.5, 90.0, 88.5, 92.0, 87.5],
        }
    )
    mock_path = tmp_dir / "mock_data.parquet"
    mock_df.to_parquet(mock_path, index=False)

    private_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "sensitive_data": ["secret1", "secret2", "secret3"],
            "value": [100, 200, 300],
        }
    )
    private_path = tmp_dir / "private_data.parquet"
    private_df.to_parquet(private_path, index=False)

    readme_path = tmp_dir / "readme.md"
    readme_path.write_text(
        "# Dataset with Parquet Files\n\nThis dataset contains parquet files."
    )

    return mock_path, private_path, readme_path


def create_test_project_folder(
    with_pyproject: bool = False,
    multiplier: int = 2,
    prefix: str = "test_project_",
) -> Path:
    """Create a test project folder with helpers package and main.py.

    Creates folder structure:
        project_dir/
        ├── pyproject.toml       # only if with_pyproject=True
        ├── main.py              # entrypoint, imports from helpers.helper
        └── helpers/
            ├── __init__.py      # package marker
            └── helper.py        # contains process_data() and get_multiplier()

    Args:
        with_pyproject: If True, creates pyproject.toml in the folder
        multiplier: Value returned by get_multiplier() in helper.py
        prefix: Prefix for the temp directory name

    Returns:
        Path to the created project directory
    """
    import tempfile

    project_dir = Path(tempfile.mkdtemp(prefix=prefix))

    # Create pyproject.toml if requested
    if with_pyproject:
        pyproject_path = project_dir / "pyproject.toml"
        pyproject_path.write_text("""
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []
""")

    # Create nested helpers package
    helpers_dir = project_dir / "helpers"
    helpers_dir.mkdir(parents=True)

    # Create __init__.py to make it a package
    init_path = helpers_dir / "__init__.py"
    init_path.write_text("# helpers package\n")

    # Create helper module
    helper_path = helpers_dir / "helper.py"
    helper_path.write_text(f'''
def process_data(data):
    """Helper function to process data."""
    return f"Processed: {{data}}"

def get_multiplier():
    return {multiplier}
''')

    # Create main.py that imports from nested helpers package
    main_path = project_dir / "main.py"
    main_path.write_text("""
import json
import syft as sy
from helpers.helper import process_data, get_multiplier

# Read data from dataset
data_path = sy.resolve_dataset_file_path("my dataset")

with open(data_path, "r") as data_file:
    data = data_file.read()

# Use helper functions
processed = process_data(data)
multiplier = get_multiplier()

result = {
    "original": data,
    "processed": processed,
    "multiplier": multiplier
}

with open("outputs/result.json", "w") as f:
    f.write(json.dumps(result))
""")

    return project_dir
