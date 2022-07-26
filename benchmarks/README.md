The benchmarking stack used by the Syft Monorepo.

Usage:

```
python executor.py MODE_OF_OPERATION SUITE_ARGS PERF_ARGS
```

Parameters:

- MODE_OF_OPERATION: selects if the executor will run the phitensor suite.
- SUITE_ARGS: control the size of the tensors for a suite of tests.
- PERF_ARGS:
  These arguments are inherited from the pyperf runner class, for more info: https://pyperf.readthedocs.io/en/latest/runner.html

### Examples

Suite args examples:

```
python executor.py --select_tests phitensor -o test.json
```

For timing purposes:

```
python executor.py --select_tests phitensor --fast -n 26 -o test.json
```

For memory allocation purposes:

```
python executor.py --select_tests phitensor --fast --tracemalloc -n 26 -o test.json
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
