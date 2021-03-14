import argparse
import os

from .app import app

parser = argparse.ArgumentParser(description="Run Infrastructure API.")

parser.add_argument(
    "--port",
    "-p",
    type=int,
    help="Port number of the API, e.g. --port=5000. Default is os.environ.get('INFRA_API_PORT', 5000).",
    default=os.environ.get("INFRA_API_PORT", 5000),
)

parser.add_argument(
    "--debug",
    "-d",
    type=bool,
    help="Debug Mode. Default is os.environ.get('INFRA_API_DEBUG', False).",
    default=os.environ.get("INFRA_API_PORT", False),
)

if __name__ == "__main__":
    args = parser.parse_args()
    app.run(port=args.port, debug=args.debug)
