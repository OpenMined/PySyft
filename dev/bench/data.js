window.BENCHMARK_DATA = {
  "lastUpdate": 1602692416877,
  "repoUrl": "https://github.com/ViveK-PothinA/PySyft",
  "entries": {
    "Python Benchmark with pytestbenchmark": [
      {
        "commit": {
          "author": {
            "email": "36106177+ramesht007@users.noreply.github.com",
            "name": "Ramesht Shukla",
            "username": "ramesht007"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "055349ee17034a6747618b04050d2f7d1a6a49bb",
          "message": "Improve tests for AST to support FSS and SNN crypto protocols (#4646)\n\n* add_tests\r\n\r\n* suggested_changes\r\n\r\n* reformatting\r\n\r\n* remove reformatting",
          "timestamp": "2020-10-14T14:42:40+02:00",
          "tree_id": "e7e1434d04f7d6a3629f9fe8ac7fa95c95a5df5b",
          "url": "https://github.com/ViveK-PothinA/PySyft/commit/055349ee17034a6747618b04050d2f7d1a6a49bb"
        },
        "date": 1602692416452,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_sigmoid",
            "value": 0.8621447923409031,
            "unit": "iter/sec",
            "range": "stddev: 0.027174142900896293",
            "extra": "mean: 1.1598979763999864 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_maclaurin",
            "value": 13.05458006495338,
            "unit": "iter/sec",
            "range": "stddev: 0.0011502437396064254",
            "extra": "mean: 76.6014682222236 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_chebyshev",
            "value": 0.8818128441922081,
            "unit": "iter/sec",
            "range": "stddev: 0.029660811074900276",
            "extra": "mean: 1.1340274828000019 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_exp",
            "value": 0.05456118074215806,
            "unit": "iter/sec",
            "range": "stddev: 0.1757681754175452",
            "extra": "mean: 18.3280491074 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_chebyshev",
            "value": 0.8521498898176922,
            "unit": "iter/sec",
            "range": "stddev: 0.040071017625196555",
            "extra": "mean: 1.1735024694000002 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}