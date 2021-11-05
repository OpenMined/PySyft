#!/bin/bash

# detect bad database version and reset db
BAD_DB_EXISTS=$(docker ps --format '{{.Names}}' | grep db | xargs -I {} docker exec {} bash -c "PGPASSWORD=$1 psql -U postgres app -c 'select version_num from alembic_version;'" | grep $2)
STATUS=$?
echo $STATUS
echo $BAD_DB_EXISTS

if [ $STATUS -eq 0 ]
then
    echo "Bad DB hash found, dropping and creating db app";
    # stop all the other containers
    OUTPUT=$(docker ps --format '{{.Names}}' | grep -v db | xargs -I {} docker stop {})
    echo $OUTPUT
    # delete the db
    OUTPUT=$(docker ps --format '{{.Names}}' | grep db | xargs -I {} docker exec {} bash -c "PGPASSWORD=$1 psql -U postgres -c 'DROP DATABASE IF EXISTS app;'")
    echo $OUTPUT
    # create the db
    OUTPUT=$(docker ps --format '{{.Names}}' | grep db | xargs -I {} docker exec {} bash -c "PGPASSWORD=$1 psql -U postgres -c 'CREATE DATABASE app;'")
    echo $OUTPUT
else
    echo "DB hash is fine, doing nothing.";
fi
exit 0
