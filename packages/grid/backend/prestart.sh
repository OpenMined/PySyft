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

if [[ "${INSTALL_OBLV_PROXY}" == "true"  &&  ("${SERVICE_NAME}" == "backend"  ||  "${SERVICE_NAME}" == "celeryworker" ) ]]; then
    echo "Allowed to install Oblv CLI"
    # Oblivious Proxy Client Installation
    mkdir -p oblv-ccli-0.4.0-x86_64-unknown-linux-musl
    tar -xf /app/wheels/oblv-ccli-0.4.0-x86_64-unknown-linux-musl.tar.gz -C oblv-ccli-0.4.0-x86_64-unknown-linux-musl
    chmod +x $(pwd)/oblv-ccli-0.4.0-x86_64-unknown-linux-musl/oblv
    ln -sf $(pwd)/oblv-ccli-0.4.0-x86_64-unknown-linux-musl/oblv /usr/local/bin/oblv  #-f is for force
    echo "Installed Oblivious CLI: $(/usr/local/bin/oblv --version)"
else
    echo "Oblivious CLI not installed INSTALL_OBLV_PROXY:${INSTALL_OBLV_PROXY} , SERVICE_NAME:${SERVICE_NAME} "
fi


# Create initial data in DB
python /app/grid/initial_data.py