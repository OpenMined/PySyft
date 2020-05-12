#!/bin/bash
exec gunicorn --reload --reload-extra-file grid/app/templates/* --graceful-timeout 0 --config dev_server.conf.py -k flask_sockets.worker "grid.app:create_app()" \
"$@"
