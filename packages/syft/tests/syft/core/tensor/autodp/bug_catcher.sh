pytest -v single_entity_phi_test.py > results.txt
if grep -q FAILED results.txt; then
  echo search is over
fi

tail results.txt | grep "FAILED"