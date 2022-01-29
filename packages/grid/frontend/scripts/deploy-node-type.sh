#!/bin/sh

cp -R /app/src/pages/_$NODE_TYPE/* /app/src/pages/
rm -rf /app/src/pages/_network /app/src/pages/_domain
