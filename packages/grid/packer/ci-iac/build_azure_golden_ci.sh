#!/bin/bash
packer init ci-golden.pkr.hcl
packer build -var-file azure_vars.json  ci-golden.pkr.hcl