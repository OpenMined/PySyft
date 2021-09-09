#! /usr/bin/env bash

pip install --user -e /app/syft

# Let the DB start
python /app/grid/backend_prestart.py

# Run migrations
alembic upgrade head

# Create initial data in DB
python /app/grid/initial_data.py

