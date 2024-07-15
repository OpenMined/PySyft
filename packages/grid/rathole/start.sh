#!/usr/bin/env bash

MODE=${MODE:-server}
RUST_LOG=${LOG_LEVEL:-trace}

# Copy configuration files
copy_config() {
  cp -L -r -f /conf/* conf/
}

# Start the server and reload until healthy
start_server() {
  while true; do
    RUST_LOG=$RUST_LOG /app/rathole conf/server.toml
    status=$?
    if [ $status -eq 0 ]; then
      break
    else
      echo "Server failed to start, retrying in 5 seconds..."
      sleep 5
    fi
  done &
}

# Start the client
start_client() {
  while true; do
    RUST_LOG=$RUST_LOG /app/rathole conf/client.toml
    status=$?
    if [ $status -eq 0 ]; then
      break
    else
      echo "Failed to load client.toml, retrying in 5 seconds..."
      sleep 10
    fi
  done &
}

# Reload configuration every 10 seconds
reload_config() {
  echo "Starting configuration reload loop..."
  while true; do
    copy_config
    sleep 10
  done
}

# Make an initial copy of the configuration
copy_config

if [[ $MODE == "server" ]]; then
  start_server
elif [[ $MODE == "client" ]]; then
  start_client
else
  echo "RATHOLE MODE is set to an invalid value. Exiting."
  exit 1
fi

# Start the configuration reload in the background to keep the configuration up to date
reload_config
