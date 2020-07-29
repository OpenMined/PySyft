"""All Network routes (REST API)."""
from flask import render_template

from .. import main_routes


@main_routes.route("/", methods=["GET"])
def index():
    """Main Page."""
    return render_template("index.html")
