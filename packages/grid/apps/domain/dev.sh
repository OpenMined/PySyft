#!/bin/bash

# enable hot reloading
export FLASK_ENV=development

# use this function as the entry point
export FLASK_APP=src/app.py:create_app

# --start_local_db
export LOCAL_DATABASE=True

# run
flask run
