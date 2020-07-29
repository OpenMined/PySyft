"""All Network routes (REST API)."""
from .. import http


@http.route("/", methods=["GET"])
def index():
    """Main Page."""
    return "Open Grid Network"
