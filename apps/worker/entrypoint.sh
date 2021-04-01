#!/bin/bash
exec poetry run gunicorn --chdir ./src -k flask_sockets.worker --bind 0.0.0.0:$PORT  wsgi:app \
"$@"
