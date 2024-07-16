#!/bin/bash

# shellcheck disable=SC2164
parent_directory=$(cd "$(dirname "$0")"; pwd)
compute_price_module_path="$parent_directory/node_pricing_structure.py"

syft launch \
  --name=test-domain-1 \
  --port=8080 \
  --reset=True \
  --dev-mode=True \
  --payment-required=True \
  --server-payment-handle=server-payment-handle \
  --payment-api=https://domain.tld/api/payments/ \
  --compute-price-module-path="$compute_price_module_path" \
  --compute-price-func-name=compute_price
