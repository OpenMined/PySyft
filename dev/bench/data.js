window.BENCHMARK_DATA = {
  "lastUpdate": 1604874786986,
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
      },
      {
        "commit": {
          "author": {
            "email": "anubhavraj.08@gmail.com",
            "name": "Anubhav Raj Singh",
            "username": "aanurraj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d2bbbc994aa7e7428d9686c3c160e3bcaff881d6",
          "message": "fix comparison between FPT and AST (#4752)\n\n* fixed comparison in AST and added tests\r\n\r\n* added seperate test cases\r\n\r\n* bugs fixed\r\n\r\n* clean\r\n\r\n* improved tests",
          "timestamp": "2020-11-05T18:29:49+05:30",
          "tree_id": "40012e289d445f25e2cbf3f4a9ee3aee07a49b0d",
          "url": "https://github.com/OpenMined/PySyft/commit/d2bbbc994aa7e7428d9686c3c160e3bcaff881d6"
        },
        "date": 1604581611802,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_exp",
            "value": 0.04164636665089897,
            "unit": "iter/sec",
            "range": "stddev: 0.16393190407046426",
            "extra": "mean: 24.011698508600013 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_sigmoid",
            "value": 0.6570560230706138,
            "unit": "iter/sec",
            "range": "stddev: 0.026808675157999607",
            "extra": "mean: 1.5219402377999813 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_chebyshev",
            "value": 0.6630106798273476,
            "unit": "iter/sec",
            "range": "stddev: 0.05985465397211117",
            "extra": "mean: 1.5082713302000001 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_maclaurin",
            "value": 10.28920131838342,
            "unit": "iter/sec",
            "range": "stddev: 0.0016376953124847332",
            "extra": "mean: 97.18927339999937 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_chebyshev",
            "value": 0.6585176439313789,
            "unit": "iter/sec",
            "range": "stddev: 0.023097390537443672",
            "extra": "mean: 1.5185621968000078 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "xvtongye1986@163.com",
            "name": "xvtongye",
            "username": "xutongye"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4575a50f38b78728dafe2615aad9145dae17b085",
          "message": "Add .ndim and .T to Tensor (OpenMined#4617) (#4773)\n\n* Add .ndim and .T to FixPrecisionTensor and PointerTensor (OpenMined#4617)\r\n\r\n* Add .ndim and .T to FixPrecisionTensor PointerTensor (OpenMined#4617)\r\n\r\n* Add .ndim .T to FixPrecisionTensor PointerTensor (OpenMined#4617)\r\n\r\n* Add .ndim .T to FixPrecisionTensor PointerTensor (OpenMined#4617)",
          "timestamp": "2020-11-09T00:26:50+02:00",
          "tree_id": "65cf8c3e8aec061339d8af1e7aaaeba956f0423c",
          "url": "https://github.com/OpenMined/PySyft/commit/4575a50f38b78728dafe2615aad9145dae17b085"
        },
        "date": 1604874786399,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_maclaurin",
            "value": 12.925640971921919,
            "unit": "iter/sec",
            "range": "stddev: 0.0025414765062902428",
            "extra": "mean: 77.36560238461502 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_chebyshev",
            "value": 0.8432242964290496,
            "unit": "iter/sec",
            "range": "stddev: 0.0406869737640078",
            "extra": "mean: 1.1859240823999926 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_exp",
            "value": 0.05530001352270804,
            "unit": "iter/sec",
            "range": "stddev: 0.29825758696914934",
            "extra": "mean: 18.08317821820001 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_tanh_sigmoid",
            "value": 0.8805129046135505,
            "unit": "iter/sec",
            "range": "stddev: 0.017556721928650404",
            "extra": "mean: 1.1357016969999905 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/frameworks/torch/mpc/pytestbenchmark/bench.py::test_sigmoid_chebyshev",
            "value": 0.8602497004033761,
            "unit": "iter/sec",
            "range": "stddev: 0.018625928397618976",
            "extra": "mean: 1.1624531801999978 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}