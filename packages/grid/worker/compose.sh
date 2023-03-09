#!/bin/bash
export STACK_API_KEY=test
export VERSION=0.8.0-beta.0
export VERSION_HASH=unknown
docker compose --env-file ../.env --file docker-compose.yml pull --ignore-pull-failures
docker compose --env-file ../.env --file docker-compose.yml --file docker-compose.build.yml build
docker compose -p worker --env-file ../.env --file docker-compose.yml --file docker-compose.dev.yml up -d
