for iteration in {1..100}
do
  echo $iteration
  pytest -v single_entity_phi_test.py > results.txt

  if grep -q FAILED results.txt; then
    echo search is over
    cp results.txt error_log
    break
  fi
done