#!/bin/bash
ls | grep ^syft- | xargs -I{} bash -c "cd {}; ./scripts/build_proto.sh; cd -"
