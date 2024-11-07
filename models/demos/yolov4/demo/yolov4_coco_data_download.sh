#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown
fi

DATA_PATH_ID="1IjLE0mDIKBgfdYReinJaBbrKe-TzY79m"
DATA_FOLDER="models/demos/yolov4/demo/fiftyone_coco_validation.zip"

gdown "https://drive.google.com/uc?id=${DATA_PATH_ID}" -O "${DATA_FOLDER}"

if ! command -v unzip &> /dev/null; then
    echo "unzip command is not available. Installing unzip..."
    sudo apt-get update
    sudo apt-get install -y unzip
fi

OUTPUT_DIRECTORY="models/demos/yolov4/demo/fiftyone_coco_validation"
mkdir -p "$OUTPUT_DIRECTORY"
unzip "$DATA_FOLDER" -d "$OUTPUT_DIRECTORY"
