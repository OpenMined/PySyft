#! /usr/bin/env bash

echo "Running prestart.sh with RELEASE=${RELEASE}"

while [ ! -f /app/syft/setup.py ]
do
    echo "Waiting for syft folder to sync"
    sleep 1
done

if [ "${RELEASE}" = "development" ]
then
    echo "Installing Syft"
    pip install --user -e /app/syft[dev]
fi

# Let the DB start
python /app/grid/backend_prestart.py

# Run migrations
cd /app && alembic upgrade head

# Create initial data in DB
python /app/grid/initial_data.py
