from flask import Blueprint

main = Blueprint("main", __name__)

from .persistence.models import db

from . import routes
