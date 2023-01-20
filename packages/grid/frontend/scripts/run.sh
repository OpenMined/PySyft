#!/bin/sh
cd /app
YARN_CACHE_FOLDER=/root/.yarn yarn install --frozen-lockfile
/usr/local/bin/node --max-old-space-size=4096 /app/node_modules/.bin/next dev -p 80
