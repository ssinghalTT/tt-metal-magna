# DEIT Device Perf computation

## Ensure to build with profiler enabled:
`build_metal.sh -p`

## Compute Device FPS
Run the following command to generate the perf sheet. FPS = (Batch_size * 10^9)/ Sum(Device kernel Duration in ns column).<br>
```
python -m tracy -p -r -v -m pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_device_OPs.py::test_deit[1]
```
```
python -m tracy -p -r -v -m pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_device_OPs.py::test_deit[2]
```
```
python -m tracy -p -r -v -m pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_device_OPs.py::test_deit[4]
```
```
python -m tracy -p -r -v -m pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_device_OPs.py::test_deit[8]
```

## To test and evaluate the end to end perf (including IO) for batch sizes 1, 2, 4, and 8 run:
Note: this is the trace + 2cq implementation and it runs 100 iterations to reports the average end 2 end FPS.<br>
We do NOT recommend running profiler with trace implmentation as trace implementation runs for 100 interations.<br>
Running each of the pytests bellow will report the FPS and inference time per different batch sizes.<br>
```
pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_perf_e2e_2cq_trace.py::test_run_deit_trace_2cq_inference[1-device_params0]
```
```
pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_perf_e2e_2cq_trace.py::test_run_deit_trace_2cq_inference[2-device_params0]
```
```
pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_perf_e2e_2cq_trace.py::test_run_deit_trace_2cq_inference[4-device_params0]
```
```
pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_inference_perf_e2e_2cq_trace.py::test_run_deit_trace_2cq_inference[8-device_params0]
```
# cifar-10 Demo
```
pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_and_torch_cifar10_inference.py
```
# visual demo on few samples
```
pytest models/demos/wormhole/deit_tiny/demo/demo_deit_ttnn_predict.py
```

# GStreamer flow
Note: GStreamer via the python script currently supports batch=1,2,4,8. Also if you export ttnn_visualizer recommended configs, it might conflict with GStreamer and generate errors. to test the GStreamer for this model with batch-size, run:<br>
`python deit.py <batch-size>`

The recommended way to test GStreamer is via plug-in command line flow which also supports batch_size>1 as follows. The gstreamer plugin environment conflicts with python gstreamer. Thus need to create a different virtual env:

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
To test the plug-in execute the following command replacing <batch-size> with 1,2, 4 or 8.
```
gst-launch-1.0 videotestsrc num-buffers=10000 pattern=black is-live=true ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=500/1 ! queue ! deit batch-size=<batch-size> !  fpsdisplaysink video-sink=fakesink -v
```
