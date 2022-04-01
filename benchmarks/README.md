The benchmarking stack used by the Syft Monorepo.

Usage:
```
python executor.py MODE_OF_OPERATION SUITE_ARGS PERF_ARGS
```
Parameters:
* MODE_OF_OPERATION: selects if the executor will run the sept suite, the rept suite or both. Choice from {sept, rept, all}.
* SUITE_ARGS: control the size of the tensors for a suite of tests.
  * --sept_rows: number of rows for the sept scenario
  * --sept_cols: number of cols for the sept scenario
  * --rept_rows: number of rows for the rept scenario
  * --rept_cols: number of cols for the rept scenario
  * --rept_dimension: dimension of the row tensor
* PERF_ARGS:
      These arguments are inherited from the pyperf runner class, for more info: https://pyperf.readthedocs.io/en/latest/runner.html

### Examples

Suite args examples:
```
python executor.py all --sept_rows=1000 --sept_cols=10 --rept_rows=1000 --rept_cols=10 --rept_dimension=15
```

For timing purposes:
```
python executor.py --fast -n 26 -o test.json
```

For memory allocation purposes:
```
python executor.py --fast --tracemalloc -n 26 -o test.json
```

For interpretion:

Statistics:
```
pyperf stats test.json
```

Histogram:
```
pyperf hist test.json
```
