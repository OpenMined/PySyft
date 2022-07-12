#!/bin/bash
(docker logs "${1}" -f &) | grep -q "${2}" || true
