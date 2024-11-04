## ttnn.to_layout fails to convert tensor on mesh device to ROW_MAJOR_LAYOUT

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_layout.py`

ttnn.to_layout fails to convert weight tensor on mesh device to ROW_MAJOR_LAYOUT whereas in single device, it works fine.

```
ttnn_input_tensor = ttnn.from_device(ttnn_input_tensor)
ttnn_input_tensor = ttnn.to_layout(ttnn_input_tensor, ttnn.ROW_MAJOR_LAYOUT)
```

ERROR MESSAGE:

```
"Device storage isn't supported"
```


Tried changing the order and checked, but it didn't work either

```
ttnn_input_tensor = ttnn.to_layout(ttnn_input_tensor, ttnn.ROW_MAJOR_LAYOUT)
ttnn_input_tensor = ttnn.from_device(ttnn_input_tensor)
```

ERROR MESSAGE:
```
# RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_op.cpp:25: ((this->output_tensor_end[-1] + 1) % 2 == 0)
#    info:
#    Can only unpad to row major tensor of even width
```
