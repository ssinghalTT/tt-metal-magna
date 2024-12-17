#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

os.chdir(os.getenv("TT_METAL_HOME"))
golden = json.load(open("tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/pgm_dispatch_golden.json", "r"))

THRESHOLD = 4
result = os.system(
    "build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch --benchmark_out_format=json --benchmark_out=bench.json"
)
if result != 0:
    print(f"Error: Test failed with error code {result}")
    sys.exit(result)

result = json.load(open("bench.json", "r"))

golden_benchmarks = {}
for benchmark in golden["benchmarks"]:
    golden_benchmarks[benchmark["name"]] = benchmark

result_benchmarks = {}
for benchmark in result["benchmarks"]:
    result_benchmarks[benchmark["name"]] = benchmark

exit_code = 0

for name, benchmark in golden_benchmarks.items():
    if name not in result_benchmarks:
        print(f"Error: Golden benchmark {name} missing from results")
        exit_code = 1
        continue
    result = result_benchmarks[benchmark["name"]]

    if "error_occurred" in benchmark:
        if "error_occurred" not in result:
            result_time = result["IterationTime"] * 1000000
            print(f"Consider adjusting baselines. Error in {name} was fixed in result (with time {result_time:.2f}us).")
        continue

    if "error_occurred" in result:
        if "error_occurred" not in benchmark:
            print(f"Error: Benchmark {name} gave unexpected error: {result['error_message']}")
            exit_code = 1
        continue

    golden_time = benchmark["IterationTime"] * 1000000
    result_time = result["IterationTime"] * 1000000
    if result_time / golden_time > (1 + THRESHOLD / 100):
        print(f"Error:Test {name} expected value {golden_time:.2f}us but got {result_time:.2f}us")
        exit_code = 1
    if golden_time / result_time > (1 + THRESHOLD / 100):
        print(
            f"Consider adjusting baselines. Test {name} got value {result_time:.2f}us but expected {golden_time:.2f}us."
        )

for name in result_benchmarks:
    if name not in golden_benchmarks:
        print(f"Error: Result benchmark {name} missing from goldens")
        exit_code = 1

if exit_code == 0:
    print("Test successful")
sys.exit(exit_code)
