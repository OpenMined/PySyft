echo "Linting!"
tox -e lint > all_errors.txt
echo "Parsing!"
echo "Total errors: "`cat all_errors.txt | grep -c "error: "`
cat all_errors.txt | grep "error: " | grep -v "node" | grep "core" > DP_errors.txt
echo "Our errors: "`cat DP_errors.txt | grep -c "error: "`

