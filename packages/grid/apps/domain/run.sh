#!/bin/bash

if [ "$APP_ENV" = "dev" ]; then
    # enable hot reloading
    export FLASK_ENV=development
    
    # use this function as the entry point
    APP_SRC=$(pwd)/src
    export FLASK_APP=${APP_SRC}/app.py:create_app
    
    # --start_local_db
    export LOCAL_DATABASE=$LOCAL_DATABASE
    
    # allow app imports from the site-packages
    export PYTHONPATH="${PYTHONPATH}:${APP_SRC}"
    
    # run
    flask run --host=0.0.0.0 --port=$PORT
fi

if [ "$APP_ENV" = "production" ]; then
    exec gunicorn --chdir ./src -k flask_sockets.worker --bind 0.0.0.0:$PORT  wsgi:app "$@"
fi
