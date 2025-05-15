#!/usr/bin/env python3

import sys
import gi
import signal
import os
import time  # Import time module for FPS calculation & processing simulation
import numpy as np
import cv2
import torch

################### Standard Imports ##################
import transformers
from datasets import load_dataset
from transformers import AutoImageProcessor
from loguru import logger
import time

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

################### TTNN Imports ##################
import ttnn
from models.demos.wormhole.deit_tiny.demo.deit_e2e_performant import DeitTrace2CQ
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_wormhole_b0, torch2tt_tensor, is_blackhole

# from models.demos.wormhole.deit_tiny.demo.deit_test_infra import create_test_infra
from models.utility_functions import (
    is_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch_random,
    profiler,
)
from models.perf.perf_utils import prep_perf_report
import ttnn


batch_size = 1
if len(sys.argv) == 2:
    batch_size = int(sys.argv[1])

############### GStreamer Pipeline Description ################
device_id = 0
device = ttnn.CreateDevice(device_id, l1_small_size=32768, trace_region_size=3211264, num_command_queues=2)
ttnn.enable_program_cache(device)
model = DeitTrace2CQ()
model.initialize_deit_trace_2cqs_inference(device, device_batch_size=batch_size)


VIDEO_CAPS_STR = "video/x-raw,format=RGB,width=224,height=224,framerate=400/1"
# VIDEO_CAPS object created inside main() after Gst.init()
FPS_UPDATE_INTERVAL_SEC = 5  # How often to print FPS average

# --- Pipeline Definitions ---
pipeline1_desc = (
    f"videotestsrc num-buffers=10000 pattern=ball is-live=true ! videoconvert ! {VIDEO_CAPS_STR} ! "
    f"appsink name=appsink emit-signals=true max-buffers=5 drop=false"
)
pipeline2_desc = (
    f"appsrc name=appsource is-live=true block=true format=time ! "
    f"{VIDEO_CAPS_STR} ! queue name=consumer_queue ! "
    # Use sync=false so fakesink consumes buffers as they arrive
    # Use dump=false as we rely on the probe for FPS info
    f"fakesink name=fakesink_consumer sync=true dump=false silent=false"
)

# --- Global variables ---
pipeline1 = None
pipeline2 = None
appsrc = None
loop = None
probe_id = 0
# --- FPS Calculation Globals (for Pad Probe) ---
inference_time = 0
inference_time_wo_io = 0
frame_count = 0
start_time = 0
last_update_time = 0
# --- Shutdown Flag ---
shutting_down = False  # Flag to signal shutdown sequence


def process_frame(sample):
    """Processes a GStreamer sample and converts it to a NumPy array."""
    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")
    format_str = structure.get_value("format")

    ret, map_info = buf.map(Gst.MapFlags.READ)
    frame_data = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3)
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
    buf.unmap(map_info)
    return frame_data, buf.pts, buf.duration  # return frame data, and the buffer presentation timestamp (pts)


# --- Appsink "new-sample" Callback ---
def on_new_sample_from_appsink(appsink, user_data):
    """Callback triggered by appsink. Includes user processing."""
    global shutting_down, model, inference_time_wo_io, inference_time, batch_size  # Access the global flag

    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK

    frame, pts, duration = process_frame(sample)

    img = frame
    ts = time.time()
    ########################## TTNN call to model #######################
    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        exit()
    if batch_size > 1:
        n, c, h, w = img.shape
        img = img.expand(batch_size, c, h, w)
    out, time_without_io = model.run_trace_2cqs_inference(img)
    te = time.time()
    inference_time = inference_time + te - ts
    inference_time_wo_io = inference_time_wo_io + time_without_io

    gst_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
    gst_buffer.pts = pts
    gst_buffer.duration = duration
    # Push the PROCESSED buffer into appsrc
    appsrc_element = user_data
    retval = appsrc_element.emit("push-buffer", gst_buffer)
    if retval != Gst.FlowReturn.OK and not shutting_down:  # Only warn if not shutting down
        print(f"Warn: push-buffer returned: {retval}", flush=True)

    return Gst.FlowReturn.OK


# --- Pad Probe Callback for FPS ---
def sink_pad_probe_cb(pad, info, user_data):
    """Callback function for the sink pad probe to calculate FPS."""
    # Access global variables needed
    global frame_count, start_time, last_update_time, FPS_UPDATE_INTERVAL_SEC, shutting_down, inference_time_wo_io, inference_time, batch_size

    # Optional: check shutting_down flag
    if shutting_down:
        return Gst.PadProbeReturn.OK  # Allow buffer to pass for clean EOS

    # Check if it's buffer data passing through the probe
    if info.type & Gst.PadProbeType.BUFFER:
        frame_count += 1
        current_time = time.monotonic()  # Use monotonic clock

        if start_time == 0:  # Initialize on first buffer
            start_time = current_time
            last_update_time = current_time

        elapsed_since_update = current_time - last_update_time
        # Print FPS update periodically
        if elapsed_since_update >= FPS_UPDATE_INTERVAL_SEC:
            elapsed_total = current_time - start_time
            if elapsed_total > 0:  # Avoid division by zero
                avg_fps = frame_count * batch_size / elapsed_total
                avg_inference_time = inference_time / frame_count
                avg_inference_time_wo_io = inference_time_wo_io / frame_count
                # This FPS measures the rate buffers arrive at fakesink (sync=false)
                # It should reflect the bottleneck rate (Python processing).
                print(
                    f"--- Pipeline Output FPS for batch {batch_size}@ {elapsed_total:.1f}s: Average={avg_fps:.2f} , Average Inference time per batch:{avg_inference_time:.4f}, Average Inference time without IO per batch:{avg_inference_time_wo_io:.4f} (Frames={frame_count})  ---",
                    flush=True,
                )
            last_update_time = current_time  # Reset timer for next interval

    # Let the buffer pass through to fakesink
    return Gst.PadProbeReturn.OK


# --- Bus Message Handler ---
def on_bus_message(bus, message, pipeline_name):
    """Handles messages from the pipeline bus (errors, EOS, warnings)."""
    global appsrc, loop, shutting_down
    msg_type = message.type
    msg_src_element = message.src
    msg_src = msg_src_element.get_name() if isinstance(msg_src_element, Gst.Element) else "unknown_src"

    # Ignore messages if shutting down, except for errors
    if shutting_down and msg_type != Gst.MessageType.ERROR:
        return True

    if msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"ERROR ({pipeline_name}/{msg_src}): {err}\n   Debug: {debug}", file=sys.stderr, flush=True)
        if loop and loop.is_running():
            loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"WARNING ({pipeline_name}/{msg_src}): {err}\n   Debug: {debug}", file=sys.stderr, flush=True)
    elif msg_type == Gst.MessageType.EOS:
        print(f"EOS received on BUS for ({pipeline_name}/{msg_src}).", flush=True)
        if pipeline_name == "Producer" and appsrc is not None and not shutting_down:
            print(">>> P1 EOS: Signaling EOS to appsrc...", flush=True)
            ret_eos = appsrc.emit("end-of-stream")
            print(f">>> appsrc.end_of_stream returned: {ret_eos}", flush=True)
        elif pipeline_name == "Consumer":
            print(">>> P2 EOS received.", flush=True)
            # Optionally stop loop when consumer ends naturally
            # if loop and loop.is_running(): loop.quit()
    return True  # Keep bus watch active


# --- Signal Handler for Ctrl+C (Improved) ---
def sigint_handler(sig, frame):
    """Handles Ctrl+C for graceful shutdown."""
    global shutting_down, loop, appsrc
    if shutting_down:  # Prevent double execution
        print("\nShutdown already initiated.", flush=True)
        return

    print("\nCtrl+C detected. Initiating shutdown...", flush=True)
    shutting_down = True  # Set flag immediately

    # Attempt to send EOS to appsrc first
    if appsrc is not None:
        print("Signaling EOS to appsrc...", flush=True)
        ret_eos = appsrc.emit("end-of-stream")
        print(f">>> appsrc.end_of_stream() returned: {ret_eos}", flush=True)
    else:
        print("appsrc not available to send EOS.", flush=True)

    # Quit the main loop directly
    if loop is not None and loop.is_running():
        print("Quitting main loop...", flush=True)
        loop.quit()
    else:
        print("Main loop not running or not found when quitting.", flush=True)


# --- Main Function ---
def main():
    global pipeline1, pipeline2, loop, appsrc, probe_id, frame_count, start_time, shutting_down

    # Initialize GStreamer FIRST
    print("Initializing GStreamer...", flush=True)
    Gst.init(sys.argv)
    print("GStreamer Initialized.", flush=True)

    # NOW create Gst specific objects
    print("Creating Gst.Caps object...", flush=True)
    VIDEO_CAPS = Gst.Caps.from_string(VIDEO_CAPS_STR)
    if not VIDEO_CAPS:
        sys.exit("FATAL: Could not parse CAPS string.")
    print("Gst.Caps object created.", flush=True)

    print("Setting up MainLoop and Signal Handler...", flush=True)
    loop = GLib.MainLoop()
    shutting_down = False  # Ensure flag is reset on start
    signal.signal(signal.SIGINT, sigint_handler)
    print("MainLoop and Signal Handler set up.", flush=True)

    # --- Create Pipelines ---
    print(f"Creating Pipeline 1: {pipeline1_desc}", flush=True)
    pipeline1 = Gst.parse_launch(pipeline1_desc)
    if not pipeline1:
        sys.exit("FATAL: Could not create Pipeline 1.")
    bus1 = pipeline1.get_bus()
    bus1.add_signal_watch()
    bus1.connect("message", on_bus_message, "Producer")
    print("Pipeline 1 created.", flush=True)

    print(f"Creating Pipeline 2: {pipeline2_desc}", flush=True)
    pipeline2 = Gst.parse_launch(pipeline2_desc)
    if not pipeline2:
        sys.exit("FATAL: Could not create Pipeline 2.")
    bus2 = pipeline2.get_bus()
    bus2.add_signal_watch()
    bus2.connect("message", on_bus_message, "Consumer")
    print("Pipeline 2 created.", flush=True)

    # --- Get appsink and appsrc elements ---
    print("Getting appsink/appsrc elements...", flush=True)
    appsink = pipeline1.get_by_name("appsink")
    appsrc = pipeline2.get_by_name("appsource")
    if not appsink or not appsrc:
        sys.exit("FATAL: Could not get appsink or appsource element.")
    print("Got appsink/appsrc elements.", flush=True)

    # --- Configure appsrc ---
    print("Configuring appsrc caps...", flush=True)
    appsrc.set_property("caps", VIDEO_CAPS)
    appsrc.set_property("format", Gst.Format.TIME)
    print("Configured appsrc.", flush=True)

    # --- Connect appsink signal ---
    print("Connecting appsink 'new-sample' signal...", flush=True)
    appsink.connect("new-sample", on_new_sample_from_appsink, appsrc)
    print("Connected appsink signal.", flush=True)

    # --- Attach Pad Probe for FPS ---
    fakesink = pipeline2.get_by_name("fakesink_consumer")
    sinkpad = None  # Keep track of sinkpad if needed later
    if fakesink:
        sinkpad = fakesink.get_static_pad("sink")
        if sinkpad:
            print("Attaching FPS probe to fakesink sink pad...", flush=True)
            probe_id = sinkpad.add_probe(Gst.PadProbeType.BUFFER, sink_pad_probe_cb, None)
            if probe_id == 0:
                print("Warning: Could not add probe to fakesink pad.", flush=True)
            else:
                print(f"Probe attached (ID: {probe_id}).", flush=True)
        else:
            print("Warning: Could not get sink pad from fakesink.", flush=True)
    else:
        print("Warning: Could not get fakesink element for FPS probe.", flush=True)

    # Reset FPS counters before starting
    frame_count = 0
    start_time = 0

    # --- Start Pipelines ---
    print("\nSetting Pipeline 1 to PLAYING...", flush=True)
    ret1 = pipeline1.set_state(Gst.State.PLAYING)
    if ret1 == Gst.StateChangeReturn.FAILURE:
        sys.exit("FATAL: Failed to set Pipeline 1 to PLAYING.")
    print("Pipeline 1 PLAYING.", flush=True)

    print("Setting Pipeline 2 to PLAYING...", flush=True)
    ret2 = pipeline2.set_state(Gst.State.PLAYING)
    if ret2 == Gst.StateChangeReturn.FAILURE:
        if pipeline1:
            pipeline1.set_state(Gst.State.NULL)
        sys.exit("FATAL: Failed to set Pipeline 2 to PLAYING.")
    print("Pipeline 2 PLAYING.", flush=True)

    print("\nPipelines running. Press Ctrl+C to stop.", flush=True)
    # Updated message
    print(
        f"Bridging via appsink/appsrc. Calculating Pipeline Output FPS every {FPS_UPDATE_INTERVAL_SEC} seconds...",
        flush=True,
    )

    # --- Run Main Loop ---
    try:
        loop.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught (should have been handled by SIGINT). Forcing shutdown.", flush=True)
        shutting_down = True
        if appsrc is not None:
            appsrc.end_of_stream()

    finally:
        # --- Cleanup ---
        print("\nMain loop finished. Cleaning up pipelines...", flush=True)
        # Stop consumer first
        if pipeline2:
            print("Setting Pipeline 2 to NULL", flush=True)
            pipeline2.set_state(Gst.State.NULL)
        # Then producer
        if pipeline1:
            print("Setting Pipeline 1 to NULL", flush=True)
            pipeline1.set_state(Gst.State.NULL)
        print("Pipelines stopped.", flush=True)


if __name__ == "__main__":
    main()
