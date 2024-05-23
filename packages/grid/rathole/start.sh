#!/usr/bin/env bash
MODE=${MODE:-server}

cp -L -r -f /conf/* conf/

if [[ $MODE == "server" ]]; then
  RUST_LOG=trace /app/rathole conf/server.toml &
elif [[ $MODE = "client" ]]; then
  while true; do
    RUST_LOG=trace /app/rathole conf/client.toml
    status=$?
    if [ $status -eq 0 ]; then
        break
    else
        echo "Failed to load client.toml, retrying in 5 seconds..."
        sleep 10
    fi
  done &
else
  echo "RATHOLE MODE is set to an invalid value. Exiting."
fi

# reload config every 10 seconds
while true
do
    # Execute your script here
    cp -L -r -f /conf/* conf/
    # Sleep for 10 seconds
    sleep 10
done