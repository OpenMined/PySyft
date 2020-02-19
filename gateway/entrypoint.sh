#!/bin/bash
exec gunicorn -k flask_sockets.worker gateway:app \
"$@"
