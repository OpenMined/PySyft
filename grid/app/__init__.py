from flask import Flask
from .config import Config
from .config import app
from .config import db
from flask_migrate import Migrate

migrate = Migrate(app, db)
