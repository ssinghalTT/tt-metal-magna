# Ultra-Fast-Lane-Detection-v2 rn18 like

### Introduction

The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

## Ensure to build with profiler enabled:
```
build_metal.sh -p
```

We currently support batch = 1 only for this model.

### Device perf
Run the following command to generate the perf sheet. FPS = (Batch_size * 10^9)/ Sum(Device kernel Duration in ns column).<br>

```
python -m tracy -p -r -v -m pytest models/experimental/ufld_v2_rn18like/tests/test_ttnn_ufld_v2_rn18like.py::test_ufld_rn18like[device_params0-pretrained_weight_false-1-3-320-800]
```

### end to end perf
```
pytest models/experimental/ufld_v2_rn18like/tests/test_ufld_v2_rn18like_e2e_performant.py
```

### Inference

- To run the demo for UFLD_v2 RN18 Model:
```bash
    pytest models/experimental/ufld_v2_rn18like/demo/inference.py
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

```bash
    pytest models/experimental/ufld_v2_rn18like/demo/demo.py
```

# GStreamer flow
Note: GStreamer via the python script currently supports batch=1. Also if you export ttnn_visualizer recommended configs, it might conflict with GStreamer and generate errors. to test the GStreamer for this model with batch-size, run:<br>
`python ufld.py`

The recommended way to test GStreamer is via plug-in command line flow which supports batch_size=1 for UFLD as follows. The gstreamer plugin environment conflicts with python gstreamer. Thus need to create a different virtual env: (If you have already created the env for gstreamer simply activate it and jump to running the plug-in command.

NOTE: The FPS displayed is for 1 pipeline. Thus the overall FPS = Gstreamer FPS * batch-size
```
deactivate		## Deactivate other environment.
./create_venv_gstreamer.sh
source python_env_gstreamer/bin/activate
```
```
sudo apt install python3-gi python3-gi-cairo
sudo apt install python3-gst-1.0 gstreamer1.0-python3-plugin-loader
sudo apt install gstreamer1.0-tools
pip install graphviz numpy_ringbuffer
sudo apt install ubuntu-restricted-extras
sudo apt-get install -y gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad gstreamer1.0-libav
```
```
export GST_PLUGIN_PATH=$PWD/plugin:$PWD/plugins
```
```
rm ~/.cache/gstreamer-1.0/registry.x86_64.bin
gst-inspect-1.0 python
```
```
OUTPUT:
Plugin Details:
  Name                     python
  Description              loader for plugins written in python
  Filename                 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstpython.so
  Version                  1.20.1
  License                  LGPL
  Source module            gst-python
  Binary package           GStreamer Python
  Origin URL               http://gstreamer.freedesktop.org

  ExampleTransform: ExampleTransform Python
  audioplot: AudioPlotFilter
  identity_py: Identity Python
  mv2like: Mv2Like Python
  mysink: CustomSink
  mytextprepender: MyTextPrepender Python
  py_audiotestsrc: CustomSrc
  py_videomixer: Videomixer
  ufld: UFLD Python

  9 features:
  +-- 9 elements
```

command to run the GStreamer plug-in: (please note: UFLD currently runs for batch=1 only)
NOTE: The FPS displayed is for 1 pipeline. Thus the overall FPS = Gstreamer FPS * batch-size
```
gst-launch-1.0 videotestsrc num-buffers=10000 pattern=black is-live=true ! videoconvert ! video/x-raw,format=RGB,width=800,height=320,framerate=500/1 ! queue ! ufld batch-size=1 !  fpsdisplaysink video-sink=fakesink -v
```
