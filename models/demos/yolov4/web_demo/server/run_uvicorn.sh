#!/bin/bash
TT_BACKEND_TIMEOUT=0 /home/dvartanians/Metal/yolov4_webdemo/tt-metal/python_env/bin/uvicorn --host 0.0.0.0 --port 7000 models.demos.yolov4.web_demo.server.fast_api_yolov4:app
