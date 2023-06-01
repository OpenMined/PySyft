# stdlib
import datetime
from pathlib import Path
import random
import string
import tempfile

temp_dir = tempfile.TemporaryDirectory().name
sqlite_workspace_folder = Path(temp_dir) / "sqlite"

test_verify_key_string_root = (
    "08e5bcddfd55cdff0f7f6a62d63a43585734c6e7a17b2ffb3f3efe322c3cecc5"
)
test_verify_key_string_client = (
    "833035a1c408e7f2176a0b0cd4ba0bc74da466456ea84f7ba4e28236e7e303ab"
)
test_verify_key_string_hacker = (
    "8f4412396d3418d17c08a8f46592621a5d57e0daf1c93e2134c30f50d666801d"
)


def generate_db_name(length: int = 10) -> str:
    random.seed(datetime.datetime.now().timestamp())
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))
