#! /usr/bin/env bash

while [ ! -f /app/syft/setup.py ]
do
    echo "Waiting for syft folder to sync"
    sleep 1
done

pip install --user -e /app/syft

# Let the DB start
python /app/grid/backend_prestart.py

# Run migrations
alembic upgrade head

# Create initial data in DB
python /app/grid/initial_data.py

