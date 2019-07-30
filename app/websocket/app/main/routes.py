"""
This file exists to provide one common place for all http requests
"""

from flask import render_template
from . import main


@main.route("/", methods=["GET"])
def index():
    return render_template("index.html")
