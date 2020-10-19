window.BENCHMARK_DATA = {
  "lastUpdate": 1603132097316,
  "repoUrl": "https://github.com/OpenMined/PySyft",
  "entries": {
    "Python Benchmark with pytestbenchmark": [
      {
        "commit": {
          "author": {
            "email": "murarugeorgec@gmail.com",
            "name": "George-Cristian Muraru",
            "username": "gmuraru"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "489972a9e467eb20a7ebbe2b5e37c63b3d5bb2aa",
          "message": "Reduce length of AST message (#4659)\n\n* Reduce length of AST message\r\n\r\n* Add serde test",
          "timestamp": "2020-10-19T21:21:57+03:00",
          "tree_id": "2c49032c68ffbf403cb3456bd9f406ff38b81c3d",
          "url": "https://github.com/OpenMined/PySyft/commit/489972a9e467eb20a7ebbe2b5e37c63b3d5bb2aa"
        },
        "date": 1603132096820,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_exp",
            "value": 0.053332067672631885,
            "unit": "iter/sec",
            "range": "stddev: 0.27892963902512324",
            "extra": "mean: 18.750444969399986 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_sigmoid",
            "value": 0.8307699868619418,
            "unit": "iter/sec",
            "range": "stddev: 0.009560454437668446",
            "extra": "mean: 1.203702608200001 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_maclaurin",
            "value": 12.454314864264312,
            "unit": "iter/sec",
            "range": "stddev: 0.0003790655417404274",
            "extra": "mean: 80.29345740000053 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_chebyshev",
            "value": 0.8564558058483747,
            "unit": "iter/sec",
            "range": "stddev: 0.03156544430210352",
            "extra": "mean: 1.1676025700000197 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_chebyshev",
            "value": 0.8658567902540187,
            "unit": "iter/sec",
            "range": "stddev: 0.03555611755453424",
            "extra": "mean: 1.1549254001999885 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}