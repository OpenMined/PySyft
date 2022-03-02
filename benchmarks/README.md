The benchmarking stack used by the Syft Monorepo.

Usage:

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
