#!/bin/bash
exec gunicorn -k flask_sockets.worker websocket_app:app \
"$@"
