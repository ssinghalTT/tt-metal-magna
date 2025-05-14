# test_gobject_type.py
import gi

gi.require_version("GObject", "2.0")  # Or appropriate Gst versions if testing Gst objects
from gi.repository import GObject

# Gst.init(None) # if you were testing a Gst.Object subclass


class MyTestObject(GObject.Object):
    __gtype_name__ = "MyTestObject"

    def __init__(self):
        super().__init__()
        print("MyTestObject instantiated")


try:
    print(f"GType for MyTestObject: {MyTestObject.get_type()}")
    obj = MyTestObject()
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"Some other error: {e}")
