"""
    All Gateway routes (REST API).
"""
from .. import main
from flask import render_template


@main.route("/", methods=["GET"])
def index():
    """ Main Page. """
    return render_template("index.html")
