import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


class Config(object):
    POSTGRES = {
        "user": "postgres",
        "pw": "password",
        "db": "grid_example_dev",
        "host": "localhost",
        "port": "5432",
    }
    SQLALCHEMY_DATABASE_URI = (
        "postgresql://%(user)s:\%(pw)s@%(host)s:%(port)s/%(db)s" % POSTGRES
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
