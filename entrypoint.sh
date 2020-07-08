#!/bin/bash
exec gunicorn -k flask_sockets.worker "grid.__main__:app" \
"$@"
