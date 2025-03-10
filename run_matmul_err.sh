#!/bin/bash

  START_ID=-2
  LOG_DIR=/localdev/$USER
  ERROR_FILE=$LOG_DIR/error_log
  STDOUT_FILE=$LOG_DIR/stdout_log
  TIME_LIMIT=100

  run_test_in_loop() {

     export ERR_FILE_PATH=$ERROR_FILE

     #The script will add 2 and start from the counter.
     #Script the one after the last successful one.
     echo $START_ID  > $ERROR_FILE

     while true
     do
        timeout $TIME_LIMIT tt-smi -r 0 || exit 1
        TT_METAL_SLOW_DISPATCH_MODE=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf
     done

  }

  run_test_in_loop > $STDOUT_FILE
