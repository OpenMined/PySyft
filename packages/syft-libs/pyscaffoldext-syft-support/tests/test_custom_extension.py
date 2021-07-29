from pathlib import Path

from pyscaffold import cli
from pyscaffold.file_system import chdir

from pyscaffoldext.syft_support.extension import SyftSupport

EXT_FLAG = SyftSupport().flag

EXPECTED_DIRS = [
    "proto",
    "scripts",
    "src",
    "src/syft_xgboost/proto",
    "src/syft_xgboost/serde",
    "tests",
]

EXPECTED_FILES = [
    "AUTHORS.md",
    "CHANGELOG.md",
    "LICENSE.txt",
    "pyproject.toml",
    "README.md",
    "setup.cfg",
    "proto/sample.proto",
    "scripts/build_proto.sh",
    "src/syft_xgboost/__init__.py",
    "src/syft_xgboost/package-support.json",
    "src/syft_xgboost/VERSION",
    "src/syft_xgboost/serde/sample.py",
    "tests/conftest.py"
    
]

def test_add_custom_extension(tmpfolder):
    args = ["-vv",EXT_FLAG,"xgboost"]
    cli.main(args)

    with chdir("syft-xgboost"):
        assert not Path("README.rst").exists()
        for path in EXPECTED_DIRS:
            assert Path(path).is_dir()
        for path in EXPECTED_FILES:
            assert Path(path).is_file()

