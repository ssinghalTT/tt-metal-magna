# Ultra-Fast-Lane-Detection-v2 rn18 like

### Introduction

The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

## Ensure to build with profiler enabled:
`build_metal.sh -p`

### Device perf
Run the following command to generate the perf sheet. FPS = (Batch_size * 10^9)/ Sum(Device kernel Duration in ns column).<br>

`python -m tracy -p -r -v -m pytest models/experimental/ufld_v2_rn18like/tests/test_ttnn_ufld_v2_rn18like.py::test_ufld_rn18like[device_params0-pretrained_weight_false-1-3-320-800]`<br>
`python -m tracy -p -r -v -m pytest models/experimental/ufld_v2_rn18like/tests/test_ttnn_ufld_v2_rn18like.py::test_ufld_rn18like[device_params0-pretrained_weight_false-2-3-320-800]`<br>
`python -m tracy -p -r -v -m pytest models/experimental/ufld_v2_rn18like/tests/test_ttnn_ufld_v2_rn18like.py::test_ufld_rn18like[device_params0-pretrained_weight_false-4-3-320-800]`

### end to end perf
To test and evaluate the end to end perf (including IO) for batch size 1 run: (batch sizes 2 and 4 currently not supported for end to end perf test)<br>
`pytest models/experimental/ufld_v2_rn18like/tests/test_ufld_v2_rn18like_e2e_performant.py`<br>

### Demo

- To run the demo for UFLD_v2 RN18 Model:(BS-4,Height-320,Width-800)
```bash
    pytest models/experimental/ufld_v2_rn18like/demo/demo.py
```

- Results will be saved in .txt files for both Reference, ttnn Models
- If you want to run the model on Custom dataset, Make sure to add input images and its corresponding truth labels in the following paths, before running the demo:

Images:

```bash
    models/experimental/ufld_v2_rn18like/demo/images
```

Labels:

```bash
    models/experimental/ufld_v2_rn18like/demo/GT_test_labels.json
```

Use the following command to run inference and see the predicted image overlayed with the detected lanes.

Input images to be predicted should go under : models/experimental/ufld_v2_rn18like/demo/predict
Predicted output images are generated under : models/experimental/ufld_v2_rn18like/demo/predict_results

`pytest models/experimental/ufld_v2_rn18like/demo/inference.py`
