# LRASPP Device Perf computation

# Build with profiler enabled:
`build_metal.sh -p`

# Compute Device FPS
Run the following command to generate the perf shee. FPS = (Batch_size * 10^9)/ Sum(Device kernel Duration in ns column)
`python -m tracy -p -r -v -m pytest models/experimental/mv2like/test/test_ttnn_mv2_like.py::test_mv2_like[1-device_params0]`
`python -m tracy -p -r -v -m pytest models/experimental/mv2like/test/test_ttnn_mv2_like.py::test_mv2_like[2-device_params0]`
`python -m tracy -p -r -v -m pytest models/experimental/mv2like/test/test_ttnn_mv2_like.py::test_mv2_like[4-device_params0]`
`python -m tracy -p -r -v -m pytest  models/experimental/mv2like/test/test_ttnn_mv2_like.py::test_mv2_like[8-device_params0]`


# To test and evaluate the end to end perf (including IO) for batch sizes 1, 2, 4, and 8 run:
# (note: this is the trace + 2cq implementation and it runs 100 iterations to reports the average end 2 end FPS)
# We do not recommend running profiler with trace implmentation as it runs for 100 interations.
# Expected end to end throughput for batch=1: ~221 FPS
`pytest models/experimental/mv2like/test/test_mv2like_e2e_performant.py::test_run_mv2like_trace_2cq_inference[1-device_params0]`
# Expected end to end throughput for batch=2: ~364 FPS
`pytest models/experimental/mv2like/test/test_mv2like_e2e_performant.py::test_run_mv2like_trace_2cq_inference[2-device_params0]`
# Expected end to end throughput for batch=4: ~488 FPS
`pytest models/experimental/mv2like/test/test_mv2like_e2e_performant.py::test_run_mv2like_trace_2cq_inference[4-device_params0]`
# Expected end to end throughput for batch=8: ~634 FPS
`pytest models/experimental/mv2like/test/test_mv2like_e2e_performant.py::test_run_mv2like_trace_2cq_inference[8-device_params0]`
