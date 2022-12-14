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

if [ "${INSTALL_OBLV_PROXY}"="true" ]
then
    echo "Allowed to install Oblv Proxy"
    # Oblivious Proxy Client Installation
    apt-get update
    apt-get -y install wget
    wget -O oblv-ccli-0.4.0-x86_64-unknown-linux-musl.tar.gz https://api.oblivious.ai/oblv-ccli/0.4.0/oblv-ccli-0.4.0-x86_64-unknown-linux-musl.tar.gz
    mkdir -p oblv-ccli-0.4.0-x86_64-unknown-linux-musl
    tar -xf oblv-ccli-0.4.0-x86_64-unknown-linux-musl.tar.gz -C oblv-ccli-0.4.0-x86_64-unknown-linux-musl
    chmod +x $(pwd)/oblv-ccli-0.4.0-x86_64-unknown-linux-musl/oblv
    ln -sf $(pwd)/oblv-ccli-0.4.0-x86_64-unknown-linux-musl/oblv /usr/local/bin/oblv  #-f is for force
fi

# Let the DB start
python /app/grid/backend_prestart.py

# Run migrations
cd /app && alembic upgrade head

# Create initial data in DB
python /app/grid/initial_data.py