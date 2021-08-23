#!/bin/bash
function nb_to_test {
    while read -r line; do
        FILE=$(basename "${line}" | cut -d '.' -f1)
        FILE_PATH=$(dirname "${line}")
        PY_FILE="$FILE_PATH/test_${FILE}.py"
        echo "Converting Notebook: $line into Test: ${PY_FILE}"
        jupyter nbconvert "${line}" --to script --output="test_${FILE}"

        echo "what is : ${PY_FILE}"
        if [ "$(uname)" == "Darwin" ]; then
            echo "Darwin"
            # note the '' for empty file on MacOS
            sed -i '' 's/^/     /' "${PY_FILE}"
        else
            echo "Linux"
            sed -i 's/^/     /' "${PY_FILE}"
        fi

        TEST_FUNC="def test_${FILE}() -> None:"
        echo "$(echo ${TEST_FUNC}; cat ${PY_FILE})" > $PY_FILE

    done
}

# delete and then recreate test files for pytest
find examples/api -name "test_*.py*" | xargs rm
find examples/api -name "*.ipynb" | grep -v ".ipynb_checkpoints" | nb_to_test
