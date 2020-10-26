window.BENCHMARK_DATA = {
  "lastUpdate": 1603719793392,
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
          "url": "https://github.com/OpenMined/PySyft/commit/f1d55314dacb8300e2bd6d3c595883c0cb5336f6"
        },
        "date": 1603200068318,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_chebyshev",
            "value": 0.8371058374301785,
            "unit": "iter/sec",
            "range": "stddev: 0.02925063381994206",
            "extra": "mean: 1.1945920757999828 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_chebyshev",
            "value": 0.8482763921188635,
            "unit": "iter/sec",
            "range": "stddev: 0.051263506141909214",
            "extra": "mean: 1.1788610519999907 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_exp",
            "value": 0.05225905011062059,
            "unit": "iter/sec",
            "range": "stddev: 0.19437782338265033",
            "extra": "mean: 19.13544157199999 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_maclaurin",
            "value": 12.579877662953313,
            "unit": "iter/sec",
            "range": "stddev: 0.000431575497002569",
            "extra": "mean: 79.492029000005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_sigmoid",
            "value": 0.8244862452404306,
            "unit": "iter/sec",
            "range": "stddev: 0.03749463073792496",
            "extra": "mean: 1.2128765103999855 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "rladhkstn8@gmail.com",
            "name": "Wansoo Kim",
            "username": "marload"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d811ef1e91e5e2c84fbbf1edf61e6983380b4d16",
          "message": "fix C408 (#4714)",
          "timestamp": "2020-10-26T15:31:40+02:00",
          "tree_id": "7db2dad963525036d86878c80a06636fc6668eae",
          "url": "https://github.com/OpenMined/PySyft/commit/d811ef1e91e5e2c84fbbf1edf61e6983380b4d16"
        },
        "date": 1603719792876,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_maclaurin",
            "value": 12.455787598056613,
            "unit": "iter/sec",
            "range": "stddev: 0.00045463175625117355",
            "extra": "mean: 80.28396374999384 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_exp",
            "value": 0.051801081478899884,
            "unit": "iter/sec",
            "range": "stddev: 0.13785062743523468",
            "extra": "mean: 19.304616263800007 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_chebyshev",
            "value": 0.8173405065642716,
            "unit": "iter/sec",
            "range": "stddev: 0.020007859500412593",
            "extra": "mean: 1.2234802899999977 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_sigmoid",
            "value": 0.8049765682566539,
            "unit": "iter/sec",
            "range": "stddev: 0.015185792705637212",
            "extra": "mean: 1.2422721845999944 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_chebyshev",
            "value": 0.8287430129408001,
            "unit": "iter/sec",
            "range": "stddev: 0.006991693104589301",
            "extra": "mean: 1.2066466737999917 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}