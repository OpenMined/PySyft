#!/bin/bash
exec gunicorn -b 0.0.0.0:${GRID_PORT:-5000} -k flask_sockets.worker "grid.__main__:app" \
"$@"
