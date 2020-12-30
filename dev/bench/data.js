window.BENCHMARK_DATA = {
  "lastUpdate": 1603220591075,
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
      },
      {
        "commit": {
          "author": {
            "email": "kevivthapion@gmail.com",
            "name": "Vivek Pothina",
            "username": "ViveK-PothinA"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f1d55314dacb8300e2bd6d3c595883c0cb5336f6",
          "message": "changed all asset to if/raise to prevent disable of assert during PYTHONOPTIMISE env (#4655)\n\n* changed all asset to if/raise to prevent disable of assert during PYTHONOPTIMISE env\r\n\r\n* minor fix for proper inverson of assert condition\r\n\r\n* Changed to AssertionError which is handled at multiple places\r\n\r\n* Changed to AssertionError which is handled at multiple places\r\n\r\n* added simple test case in test_string to increase test coverage\r\n\r\n* added simple test cases in test_string to increase test coverage\r\n\r\n* added simple test cases to increase test coverage\r\n\r\n* Either None OR More than One worker result found\r\n\r\n* changes for review comments\r\n\r\n* removed comments, minor changes\r\n\r\nCo-authored-by: Vivek Pothina <vivek.pothina@ninjacart.com>",
          "timestamp": "2020-10-20T16:14:40+03:00",
          "tree_id": "11bfbaa619e44305fcae4d76769df1865f5e6f59",
          "url": "https://github.com/ViveK-PothinA/PySyft/commit/f1d55314dacb8300e2bd6d3c595883c0cb5336f6"
        },
        "date": 1603220590562,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_chebyshev",
            "value": 0.5991718760179439,
            "unit": "iter/sec",
            "range": "stddev: 0.02039695020250229",
            "extra": "mean: 1.6689701904000116 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_maclaurin",
            "value": 9.06871993541254,
            "unit": "iter/sec",
            "range": "stddev: 0.006730445416505764",
            "extra": "mean: 110.26914571427986 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_chebyshev",
            "value": 0.5956622823755252,
            "unit": "iter/sec",
            "range": "stddev: 0.029064406736664994",
            "extra": "mean: 1.6788036268000042 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_sigmoid",
            "value": 0.5911115489629327,
            "unit": "iter/sec",
            "range": "stddev: 0.026166565507494164",
            "extra": "mean: 1.6917280702000084 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_exp",
            "value": 0.037389675788398684,
            "unit": "iter/sec",
            "range": "stddev: 0.1261059215451701",
            "extra": "mean: 26.745350926799993 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}