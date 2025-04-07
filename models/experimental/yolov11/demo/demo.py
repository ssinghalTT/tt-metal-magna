# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import cv2
import sys
import ttnn
import torch
import pytest
import torch.nn as nn
import time
from loguru import logger
from datetime import datetime
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.yolov11.reference import yolov11
from models.experimental.yolov11.reference.yolov11 import attempt_load
from models.experimental.yolov11.tt import ttnn_yolov11
from models.experimental.yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.experimental.yolov11.demo.demo_utils import LoadImages, preprocess, postprocess

try:
    sys.modules["ultralytics"] = yolov11
    sys.modules["ultralytics.nn.tasks"] = yolov11
    sys.modules["ultralytics.nn.modules.conv"] = yolov11
    sys.modules["ultralytics.nn.modules.block"] = yolov11
    sys.modules["ultralytics.nn.modules.head"] = yolov11
except KeyError:
    print("models.experimental.yolov11.reference.yolov11 not found.")


def save_yolo_predictions_by_model(result, save_dir, image_path, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_name == "torch_model":
        bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
    else:
        bounding_box_color, label_color = (255, 0, 0), (255, 0, 0)

    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    names = result["names"]

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {score.item():.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), bounding_box_color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"prediction_{timestamp}.jpg"
    output_path = os.path.join(model_save_dir, output_name)
    cv2.imwrite(output_path, image)
    print(f"Predictions saved to {output_path}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source, model_type,resolution",
    [
        # 224*224
        ("models/experimental/yolov11/demo/images/cycle_girl.jpg", "torch_model", [3, 224, 224]),
        ("models/experimental/yolov11/demo/images/cycle_girl.jpg", "tt_model", [3, 224, 224]),
        ("models/experimental/yolov11/demo/images/dog.jpg", "torch_model", [3, 224, 224]),
        ("models/experimental/yolov11/demo/images/dog.jpg", "tt_model", [3, 224, 224]),
        # 640*640
        ("models/experimental/yolov11/demo/images/cycle_girl.jpg", "torch_model", [3, 640, 640]),
        ("models/experimental/yolov11/demo/images/cycle_girl.jpg", "tt_model", [3, 640, 640]),
        ("models/experimental/yolov11/demo/images/dog.jpg", "torch_model", [3, 640, 640]),
        ("models/experimental/yolov11/demo/images/dog.jpg", "tt_model", [3, 640, 640]),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_demo(device, source, model_type, resolution, use_pretrained_weight):
    disable_persistent_kernel_cache()
    model = yolov11.YoloV11()

    if use_pretrained_weight:
        state_dict = attempt_load("yolov11n.pt", map_location="cpu").state_dict()
    else:
        state_dict = model.state_dict()

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2

    model.load_state_dict(new_state_dict)
    if model_type == "torch_model":
        logger.info("Inferencing using Torch Model")
        model.eval()
    else:
        torch_input, ttnn_input = create_yolov11_input_tensors(
            device, input_channels=resolution[0], input_height=resolution[1], input_width=resolution[2]
        )
        parameters = create_yolov11_model_parameters(model, torch_input, device=device)
        model = ttnn_yolov11.YoloV11(device, parameters)
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/experimental/yolov11/demo/runs"
    dataset = LoadImages(path=source)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "TV",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, resolution)
        if model_type == "torch_model":
            t0 = time.time()
            preds = model(im)
            t1 = time.time()
        else:
            img = torch.permute(im, (0, 2, 3, 1))
            img = img.reshape(
                1,
                1,
                img.shape[0] * img.shape[1] * img.shape[2],
                img.shape[3],
            )
            ttnn_im = ttnn.from_torch(img, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
            t0 = time.time()
            preds = model(x=ttnn_im)
            t1 = time.time()
            preds = ttnn.to_torch(preds, dtype=torch.float32)

        print("****************************************")
        print(f"Model type: {model_type}")
        print(f"Resolution: {resolution}")
        print(f"Time taken for inference: {t1 - t0}")
        print("****************************************")
        results = postprocess(preds, im, im0s, batch, names)[0]
        save_yolo_predictions_by_model(results, save_dir, source, model_type)

    print("Inference done")
