## This commit contains Conv1d unit test for Whisper model with data parallel implementation.

1. `test_whisper_dp_conv1d` method in `tests/ttnn/unit_tests/operations/test_conv1d.py` file will contain unit test for ttnn.conv2d.

To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_conv1d.py::test_whisper_dp_conv1d`

## Expected Behaviour / Error(s):

    Expected to pass the testcase.

## Details:

#### On WH(n300):
Weights Shape - (512, 80, 3)

Case 1: Weights are not Tilized.

```
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op.cpp:76: b.get_layout() == Layout::TILE
E       info:
E       Weights should be in TILE layout.
```

Case 2: Weights are Tilized.
```
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp:371: output_channels <= b.get_legacy_shape()[3]
E       info:
E       Invalid weight shape. Incorrect weight tensor.
```
