#!/bin/bash

# enable hot reloading
export FLASK_ENV=development

# use this function as the entry point
APP_SRC=$(pwd)/src
export FLASK_APP=${APP_SRC}/app.py:create_app

# --start_local_db
export LOCAL_DATABASE=True

# allow domain imports from the site-packages
export PYTHONPATH="${PYTHONPATH}:${APP_SRC}"

# run
flask run --host=0.0.0.0
