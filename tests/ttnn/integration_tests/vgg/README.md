## Issue in creating trace+2cq test perf for vgg model as the input to the model and ttnn.Conv2d is expected to be in ttnn but on host

Runt the following command:
`pytest tests/ttnn/integration_tests/vgg/test_ttnn_vgg11.py`

Trace+2cq requires all input tensors to be on the device ready before model execution begins. However,for VGG model for ttnn.Conv2d when kept the input tensor on device throws the follwing error:

```
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/data_movement/data_transfer/data_transfer.cpp:23: input_tensor.get_legacy_shape()[-1] * input_tensor.element_size() % sizeof(uint32_t) == 0
E       info:
E       Error
E       backtrace:
```
