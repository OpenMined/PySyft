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

# Oblivious Proxy Client Installation
apt-get update
apt-get -y install wget
apt-get -y install unzip
wget -O oblv-ccli-0.3.0-x86_64-unknown-linux-musl.zip https://oblv-cli-binary.s3.us-east-2.amazonaws.com/0.3.0/oblv-ccli-0.3.0-x86_64-unknown-linux-musl.zip
unzip -o oblv-ccli-0.3.0-x86_64-unknown-linux-musl.zip
cd oblv-ccli-0.3.0-x86_64-unknown-linux-musl/
chmod +x $(pwd)/oblv
ln -sf $(pwd)/oblv /usr/local/bin/oblv  #-f is for force

# Let the DB start
python /app/grid/backend_prestart.py

# Run migrations
cd /app && alembic upgrade head

# Create initial data in DB
python /app/grid/initial_data.py