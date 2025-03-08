#!/bin/bash
  LOG_DIR=/localdev/amahmud
  LAST_HANG_F=$LOG_DIR/last_hang
  export LAST_HANG_FILE=$LAST_HANG_F
  echo "-1" > $LAST_HANG_F
  TOTAL_TESTS=125000
  TIME_LIMIT=30

  while true
  do
     timeout $TIME_LIMIT tt-smi -r 0 || exit 1
     pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf
  done
