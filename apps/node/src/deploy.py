import os
import sys

# Add dependencies in EFS to python-path
sys.path.append(os.environ.get("MOUNT_PATH"))

from app import create_lambda_app

app = create_lambda_app(node_id="bob")
