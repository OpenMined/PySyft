#!/usr/bin/env bash
MODE=${MODE:-server}

cp -L -r -f /conf/* conf/

if [[ $MODE == "server" ]]; then
  /app/rathole conf/server.toml &
elif [[ $MODE = "client" ]]; then
  /app/rathole conf/client.toml &
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