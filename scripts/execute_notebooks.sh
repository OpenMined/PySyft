# Generating notebooks
for nb in examples/*ipynb; do
    # do not run on client or server notebooks for now
    if [[ $nb = *"Client"* ]] || [[ $nb = *"Server"* ]]; then
      echo "Skipping client and server tests on $nb"
    else
        jupyter nbconvert --ExecutePreprocessor.timeout=3600 --execute "$nb" --to markdown |& tee nb_to_md.txt
        traceback=$(grep "Traceback (most recent call last):" nb_to_md.txt)
        if [[ $traceback ]]; then
            exit 1
        fi
    fi
done
