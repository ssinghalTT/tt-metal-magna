#!/usr/bin/env python3

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase

# Initialize GStreamer (only once)
Gst.init(None)


# --- Element Class Definition ---
class MyTextPrepender(GstBase.BaseTransform):
    # Element metadata (for GStreamer)
    __gstmetadata__ = (
        "MyTextPrepender Python",  # Long name
        "Filter/Effect/Converter",  # Classification
        "Prepends a configurable string to text buffer data",  # Description
        "Your Name <your.email@example.com>",  # Author
    )

    # Pad Templates
    _sink_template = Gst.PadTemplate.new("sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.new_any())
    _src_template = Gst.PadTemplate.new("src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any())
    __gsttemplates__ = (_src_template, _sink_template)  # Order can matter for some tools

    # Properties
    # To make properties work correctly with GObject.type_register and __gstelementfactory__,
    # they are best defined using GObject.Property
    # For simplicity in this direct adaptation, let's initialize them in __init__
    # and assume they are set programmatically or via default.
    # A more robust way would be to define them via GObject.Property if using this style.
    # However, __gproperties__ might still work if GObject.type_register is smart enough.
    # Let's try with __gproperties__ first as it's common.

    __gproperties__ = {
        "prefix": (
            GObject.TYPE_STRING,  # type
            "Text Prefix",  # nick
            "The string to prepend to the data.",  # blurb
            "DEFAULT: ",  # default value
            GObject.ParamFlags.READWRITE,  # flags
        )
    }

    def __init__(self):
        super().__init__()
        # Initialize properties from defaults if not set otherwise
        self.prefix = self.__gproperties__["prefix"][3]  # Default value

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "prefix":
            return self.prefix
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == "prefix":
            self.prefix = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_transform(self, inbuf: Gst.Buffer, outbuf: Gst.Buffer) -> Gst.FlowReturn:
        try:
            success, in_map_info = inbuf.map(Gst.MapFlags.READ)
            if not success:
                Gst.error("MyTextPrepender: Failed to map input buffer")
                return Gst.FlowReturn.ERROR

            original_data = in_map_info.data
            prefix_bytes = self.prefix.encode("utf-8")
            transformed_data = prefix_bytes + original_data

            inbuf.unmap(in_map_info)

            success, out_map_info = outbuf.map(Gst.MapFlags.WRITE)
            if not success:
                Gst.error("MyTextPrepender: Failed to map output buffer for writing")
                return Gst.FlowReturn.ERROR

            if len(transformed_data) > out_map_info.maxsize:
                # This is a simple handler; a real plugin might reallocate or have other strategies
                Gst.warning(
                    f"MyTextPrepender: Transformed data ({len(transformed_data)} bytes) larger than outbuf ({out_map_info.maxsize} bytes). Truncating."
                )
                out_map_info.data[:] = transformed_data[: out_map_info.maxsize]
                outbuf.props.size = out_map_info.maxsize
            else:
                out_map_info.data[: len(transformed_data)] = transformed_data  # Slice assignment
                outbuf.props.size = len(transformed_data)

            outbuf.unmap(out_map_info)
            return Gst.FlowReturn.OK

        except Exception as e:
            Gst.error(f"MyTextPrepender: Error in transform: {e}")
            if "in_map_info" in locals() and in_map_info.buffer.is_mapped():
                in_map_info.buffer.unmap(in_map_info)
            if "out_map_info" in locals() and out_map_info.buffer.is_mapped():
                out_map_info.buffer.unmap(out_map_info)
            return Gst.FlowReturn.ERROR


# --- GObject Type Registration ---
# This explicitly registers the Python class with the GObject type system.
# After this call, MyTextPrepender.get_type() would be available.
GObject.type_register(MyTextPrepender)

# --- GStreamer Element Factory ---
# This tuple is what GStreamer's Python plugin loader looks for
# to make the element available.
# ("factory-name", rank, PythonClass)
# The factory-name should ideally be unique and all lowercase.
__gstelementfactory__ = ("mytextprepender", Gst.Rank.NONE, MyTextPrepender)

# The following are not strictly needed for __gstelementfactory__ but good for plugin tools
GST_PLUGIN_NAME = "mytextprependerplugin"
__gstplugininit__ = None  # Not using a plugin_init function with __gstelementfactory__
# Metadata for the plugin file itself (distinct from element metadata)
# This isn't formally used by __gstelementfactory__ in the same way as Gst.Plugin.register_static
# but can be good practice to include. For a pure __gstelementfactory__ plugin, GStreamer
# mainly cares about that tuple.
# However, gst-inspect-1.0 might pick up __gstmetadata__ if it's at the top level of the plugin file.
# For consistency with your working example, let's keep it minimal for now.
# If you want full plugin metadata to show with gst-inspect on the *plugin file*,
# the Gst.Plugin.register_static approach with plugin_init is more standard.
# But for just getting the element to work, __gstelementfactory__ is simpler.
