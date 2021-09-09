#!/bin/sh

if [[ -z "$PROD_ROOT" ]]
then
  echo "You need to set PROD_ROOT and NODE_TYPE before building"
  exit 1
fi

cp -R $PROD_ROOT/src/pages/_$NODE_TYPE/* $PROD_ROOT/src/pages/
rm -rf $PROD_ROOT/src/pages/_network $PROD_ROOT/src/pages/_domain

