justfile_dir := justfile_directory()

_cyan := '\033[0;36m'
_red := '\033[0;31m'
_green := '\033[0;32m'
_nc := '\033[0m'

# ---------------------------------------------------------------------------------------------------------------------
# Aliases

alias b := build
alias p := publish
alias bp:= bump-and-publish
# ---------------------------------------------------------------------------------------------------------------------


test-unit:
    #!/bin/bash
    uv run pytest -n auto ./tests/unit

test-unit-perm:
    #!/bin/bash
    uv run pytest -n auto ./packages/syft-perms/tests

test-unit-permissions:
    #!/bin/bash
    uv run pytest -n auto ./packages/syft-permissions/tests

test-unit-job:
    #!/bin/bash
    uv run pytest -v ./packages/syft-job/tests


test-unit-enclave:
    #!/bin/bash
    uv run pytest -n auto ./packages/syft-enclave/tests

test-unit-fast:
    #!/bin/bash
    uv run pytest ./tests/unit --ignore=tests/unit/test_job_auto_approval.py --ignore=tests/unit/test_version_mismatch_flow.py --ignore=tests/unit/syft_bg/test_email_auto_approve_flow.py --ignore=tests/unit/syft_bg/test_email_approval_flow.py --ignore=tests/unit/test_sync_file_lock.py -k "not (test_jobs or job_flow_with_dataset)"


test-integration-mock-mode:
    #!/bin/bash
    INTEGRATION_TEST_MOCK_MODE=true uv run pytest -n auto ./tests/integration/with_unit_coverage

test-integration:
    #!/bin/bash
    uv run pytest -s ./tests/integration


test-integration-with-unit-coverage:
    #!/bin/bash
    uv run pytest -s ./tests/integration/with_unit_coverage


test-integration-without-unit-coverage:
    #!/bin/bash
    uv run pytest -s ./tests/integration/without_unit_coverage


benchmark:
    #!/bin/bash
    python ./benchmarks/benchmark_loadtime.py


# Delete syftbox for a single account (DO, DS, or enclave)
# Usage: just delete-syftbox user@example.com do
#        just delete-syftbox user@example.com enclave
delete-syftbox email name="do":
    #!/bin/bash
    set -e
    token="./credentials/token_{{name}}.json"
    [ -f "$token" ] || { echo "Error: $token not found" >&2; exit 1; }
    echo "Deleting syftbox for {{email}}..."
    uv run python -c "
    from syft_client.sync.utils.syftbox_utils import delete_syftbox
    delete_syftbox(token_path='$token', email='{{email}}')
    "

clean:
    #!/bin/sh
    printf "{{ _cyan }}Cleaning up...{{ _nc }}\n"

    # Function to remove directories by name pattern
    remove_dirs() {
        dir_name=$1
        count=$(find . -type d -name "$dir_name" 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            printf "  {{ _red }}✗{{ _nc }} Removing %s %s directories\n" "$count" "$dir_name"
            find . -type d -name "$dir_name" -exec rm -rf {} + 2>/dev/null || true
        fi
    }

    remove_dirs "syft_client.egg-info"
    remove_dirs "__pycache__"
    remove_dirs ".pytest_cache"

    printf "{{ _green }}✓ Clean complete!{{ _nc }}\n"


# Bump version (patch, minor, or major)
[group('version')]
bump part="patch":
    uvx bump2version --allow-dirty {{ part }}

# Show current version
[group('version')]
version:
    @python3 -c "import syft_client; print(syft_client.__version__)"

# Build syft client wheel
[group('build')]
build:
    @echo "{{ _cyan }}Building syft-client wheel...{{ _nc }}"
    rm -rf dist/
    uv build
    @echo "{{ _green }}Build complete!{{ _nc }}"

# Publish to PyPI
[group('publish')]
publish: build
    @echo "{{ _cyan }}Publishing to PyPI...{{ _nc }}"
    uvx twine upload dist/*
    @echo "{{ _green }}Publish complete!{{ _nc }}"

# Bump version and publish to PyPI
[group('publish')]
bump-and-publish part="patch":
    just bump {{ part }}
    just publish
    @echo "{{ _green }}Bump and publish complete!{{ _nc }}"

# Launch Jupyter Lab
jupyter:
    uv run jupyter lab --notebook-dir={{ justfile_dir }}

