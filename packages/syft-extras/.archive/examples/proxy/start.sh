#!/bin/ash

# Path to your Nginx configuration file or directory
CONFIG_PATH="/etc/nginx/nginx.conf"
# Optional: Monitor the entire directory for changes
# CONFIG_PATH="/etc/nginx/"

# Run an infinite loop to monitor the configuration file
while true; do
    # Monitor for modify, move, create, or delete events on the config file
    inotifywait -e modify,move,create,delete $CONFIG_PATH

    # Reload Nginx configuration
    echo "Nginx configuration changed. Reloading..."
    nginx -s reload
done
