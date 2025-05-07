## Build and Test Instructions

### To set up and verify the model functionality, follow these steps:

## Details

## Supported batch sizes: 1,2,3,4,8

The entry point to model is Mv2Like in `models/experimental/mv2like/tt/ttnn_mv2_like.py`.

Use the following command to run the model :
`pytest  models/experimental/mv2like/test/test_ttnn_mv2_like.py`

Use the following command to run the e_2_e perf(7 FPS):
`pytest models/experimental/mv2like/test/test_perf_mv2like.py::test_mv2like`

Use the following command to run the trace_2cq :
`pytest models/experimental/mv2like/test/test_mv2like_performant.py::test_run_mv2like_trace_2cq_inference`

Use the following command to run the model e2e perf with trace (270 FPS):
`pytest models/experimental/mv2like/test/test_mv2like_e2e_performant.py::test_run_mv2like_trace_2cq_inference`
