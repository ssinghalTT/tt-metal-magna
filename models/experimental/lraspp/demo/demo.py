# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0from PIL import Image

import os
import torch
import ttnn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import evaluate
import pytest
from loguru import logger
from torchvision import transforms
from tests.ttnn.utils_for_testing import assert_with_pcc
from PIL import Image
import torchvision.transforms.functional as TF
import cv2


from models.experimental.lraspp.reference.lraspp import LRASPP
from models.experimental.lraspp.reference.lraspp_magna import lraspp_mobilenet_v2
from models.experimental.lraspp.tt import ttnn_lraspp
from tqdm import tqdm
from models.experimental.lraspp.tt.model_preprocessing import (
    create_lraspp_model_parameters,
    get_fire_dataset_transform,
)


class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_folder, mask_folder, image_processor):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.mask_files = sorted(os.listdir(mask_folder))
        self.image_processor = get_fire_dataset_transform()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        # Pass both image and mask to the transform if required
        inputs, _ = self.image_processor(
            TF.to_tensor(cv2.imread(str(image_path))), TF.to_tensor(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
        )
        mask_np = np.array(mask)
        return {"input": inputs, "gt_mask": mask_np, "path": image_path}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_demo_semantic_segmentation(device):
    weights_path = (
        "models/experimental/lraspp/lraspp_mobilenet_v2_trained_statedict.pth"  # specify your weights path here
    )

    state_dict = torch.load(weights_path)

    magna_model = lraspp_mobilenet_v2(num_classes=1)
    magna_model.load_state_dict(state_dict)
    magna_model.eval()

    new_state_dict = {}
    torch_model = LRASPP()

    for keyname, (dict_key_name, parameters) in zip(torch_model.state_dict().keys(), state_dict.items()):
        new_state_dict[keyname] = parameters

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    image_processor = CustomImageProcessor(size=(224, 224))

    image_folder = "models/experimental/lraspp/demo/images"
    mask_folder = "models/experimental/lraspp/demo/annotations"

    dataset = SemanticSegmentationDataset(image_folder, mask_folder, image_processor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    ref_metric = evaluate.load("mean_iou")
    ttnn_metric = evaluate.load("mean_iou")
    magna_metric = evaluate.load("mean_iou")
    model_parameters = create_lraspp_model_parameters(torch_model, device=device)
    ttnn_model = ttnn_lraspp.TtLRASPP(model_parameters, device, batchsize=1)
    num_labels = None
    pccs = []
    os.makedirs("models/experimental/lraspp/demo/ttnn_results", exist_ok=True)
    for batch in tqdm(data_loader, desc="Processing batches"):
        image = batch["input"]
        mask = batch["gt_mask"].squeeze()
        path = batch["path"]
        filename = path[0].split("/")[-1]
        print("num_labels", num_labels)
        torch_input_tensor_permuted = torch.permute(image, (0, 2, 3, 1))
        torch_input_tensor = image
        n, c, h, w = torch_input_tensor.shape
        if c == 3:
            c = 16
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=7),
            ttnn.ShardStrategy.HEIGHT,
        )
        ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn_input_tensor = ttnn_input_tensor.to(device, input_mem_config)

        print("the shape of the tensor is", ttnn_input_tensor.shape)
        ttnn_output = ttnn_model(ttnn_input_tensor)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_final_output = ttnn_output[:, :, :, 0:1].permute(0, 3, 1, 2)

        ref_logits = torch_model(image)
        magna_logits = magna_model(image)["out"]

        ref_upsampled_logits = torch.nn.functional.interpolate(
            ref_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
        ttnn_upsampled_logits2 = torch.nn.functional.interpolate(
            ttnn_final_output, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
        magna_upsampled_logits = torch.nn.functional.interpolate(
            magna_logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
        
        thres = 0.5
        ref_pred_mask = (ref_upsampled_logits > thres).squeeze().cpu().numpy()
        ttnn_pred_mask = (ttnn_upsampled_logits2 > thres).squeeze().cpu().numpy()
        magna_pred_mask = (magna_upsampled_logits > thres).squeeze().cpu().numpy()

        pcc = assert_with_pcc(ref_pred_mask, ttnn_pred_mask, 0.1)  # masks are identical
        pccs.append(pcc[-1])

        ref_pred_mask_np = ref_pred_mask.astype(np.uint8)
        ttnn_pred_mask_np = ttnn_pred_mask.astype(np.uint8)
        magna_pred_mask_np = magna_pred_mask.astype(np.uint8)

        ref_image = Image.fromarray(ref_pred_mask_np, mode="L")
        ttnn_image = Image.fromarray(ttnn_pred_mask_np, mode="L")
        magna_image = Image.fromarray(magna_pred_mask_np, mode="L")

        # Load the original image
        orig_image = cv2.imread(path[0])
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_image.shape[:2]

        # Resize masks to original image size
        ref_pred_mask_resized = cv2.resize(ref_pred_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        ttnn_pred_mask_resized = cv2.resize(ttnn_pred_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        magna_pred_mask_resized = cv2.resize(magna_pred_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_np = np.array(mask)
        mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Prepare overlays: only overlay where mask==1 (fire)
        mask_fire = mask_resized == 255  # Assuming 255 is the fire label in the mask
        ref_fire = ref_pred_mask_resized == 1
        ttnn_fire = ttnn_pred_mask_resized == 1
        magna_fire = magna_pred_mask_resized == 1

        # Create overlay images (copy to avoid modifying original)
        overlay_mask = orig_image.copy()
        overlay_ref = orig_image.copy()
        overlay_ttnn = orig_image.copy()
        overlay_mag = orig_image.copy()

        # Overlay: mask in red, ref in green, ttnn in blue
        overlay_mask[mask_fire] = [128, 0, 128]  # Red for GT mask
        overlay_ref[ref_fire] = [0, 255, 0]  # Green for ref pred
        overlay_ttnn[ttnn_fire] = [0, 0, 255]  # Blue for ttnn pred
        overlay_mag[magna_fire] = [255, 165, 0]  # orange for magna pred

        # Stack overlays horizontally
        combined = np.concatenate([overlay_mask, overlay_ref, overlay_mag, overlay_ttnn], axis=1)

        # Add labels for each overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        label_colors = [(128, 0, 128), (0, 255, 0), (255, 165, 0), (0, 0, 255)]  # White text

        labels = ["Purple: GT Mask", "Green: Torch", "Orange: Magna", "Blue: TTNN"]
        label_y = 30
        width = orig_image.shape[1]
        for i, label in enumerate(labels):
            x = i * width + 10
            cv2.putText(combined, label, (x, label_y), font, font_scale, label_colors[i], thickness, cv2.LINE_AA)

        # Save the combined image
        save_path = os.path.join("models/experimental/lraspp/demo/ttnn_results", f"overlay_{filename}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(combined).save(save_path)

        mask = mask_resized
        ref_metric.add(predictions=ref_pred_mask_resized, references=mask)
        ttnn_metric.add(predictions=ttnn_pred_mask_resized, references=mask)
        magna_metric.add(predictions=magna_pred_mask_resized, references=mask)

        ref_results = ref_metric.compute(num_labels=2, ignore_index=255, reduce_labels=False)
        ttnn_results = ttnn_metric.compute(num_labels=2, ignore_index=255, reduce_labels=False)
        magna_results = magna_metric.compute(num_labels=2, ignore_index=255, reduce_labels=False)

        logger.info(
        f"mean IoU values for Reference and ttnn model and Magna reference are {ref_results['mean_iou']}, {ttnn_results['mean_iou']}, {magna_results['mean_iou']} respectively"
        )
        logger.info(f"mean PCC values between Reference and ttnn model are {np.mean(pccs)}")


class CustomImageProcessor:
    def __init__(self, size=(512, 512), mean=None, std=None):
        self.size = size
        self.mean = mean if mean else [0.485, 0.456, 0.406]
        self.std = std if std else [0.229, 0.224, 0.225]
        self.transform = transforms.Compose(
            [transforms.Resize(self.size), transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )

    def __call__(self, image, return_tensors="pt"):
        processed_image = self.transform(image)
        if return_tensors == "pt":
            return {"pixel_values": processed_image.unsqueeze(0)}
        elif return_tensors == "np":
            return {"pixel_values": processed_image.numpy().transpose(1, 2, 0)}
        else:
            raise ValueError("Unsupported return_tensors format. Use 'pt' or 'np'.")
