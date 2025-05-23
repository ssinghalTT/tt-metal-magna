from pathlib import Path
import sys
from typing import List, Optional, Tuple

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, random_split
#from torchvision.transforms import v2
from models.experimental.lraspp.tt.seg_transforms import Compose


class BinarySegmentationSet(Dataset):
    def __init__(self, data_dir: Path, mask_dir: Path, transform: Optional[Compose] = None, image_ext: str = 'jpg') -> None:
        super().__init__()
        self._images = []
        self._masks = []
        self._transform = transform
        if not data_dir.exists():
            print("[ERROR] Path does not exists")
            sys.exit()

        # cnt = 0
        for image, mask in zip(data_dir.glob(f'*.{image_ext.lower()}'), mask_dir.glob(f'*.{image_ext.lower()}')):
            image = cv2.imread(str(image))
            mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            self._images.append(TF.to_tensor(image))
            self._masks.append(TF.to_tensor(mask))

            # if cnt >= 100:
            #     break
            # cnt += 1

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        _image = self._images[index]
        _mask = self._masks[index]

        if self._transform:
            # image = self._transform(image)
            # mask = self._transform(mask)

            image, mask = self._transform(_image, _mask)
        return image, mask


class BinarySegmentationData:
    def __init__(
        self,
        train_dir: List[str],
        mask_dir: List[str],
        val_dir: Optional[str] = None,
        train_transform: Optional[Compose] = None,
        val_transform: Optional[Compose] = None,
    ) -> None:
        train_dir = [Path(dd) for dd in train_dir]
        mask_dir = [Path(dd) for dd in mask_dir]

        if not val_dir:
            self.dataset = BinarySegmentationSet(
                data_dir=train_dir[0],
                mask_dir=mask_dir[0],
                transform=train_transform)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [0.8, 0.2])
        else:
            val_dirs = Path(val_dir)
            # TODO: setup val_dir

    def split(self) -> Tuple[Dataset, Dataset]:
        return self.train_dataset, self.val_dataset

    def __repr__(self) -> str:
        return (
            "BinarySegmentationData for simulation\n"
            + f"\t samples for training:\t{len(self.train_dataset)}\n"
            + f"\t samples for training:\t{len(self.val_dataset)}\n"
        )


if __name__ == "__main__":
    from seg_transforms import (
        CenterCrop,
        ColorJitter,
        RandomCrop,
        RandomHorizontalFlip,
        RandomRotation,
        RandomScaledResize,
        RandomVerticalFlip,
    )

    train_transform = Compose(
        [
            ColorJitter(brightness=0.25, contrast=0.1,
                        saturation=0.1, hue=0.1),
            RandomScaledResize((0.9, 1.1)),
            RandomRotation((0, 90)),
            RandomCrop((256, 160)),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
        ]
    )
    # val_transform = Compose([CenterCrop(size=(256, 160))])

    # change this to your local path where you stored the fire dataset downloaded from e.g. kaggle!
    #dataset_path = "/home/vivepat4/datasets/fire_dataset/Fire"
    #mask_path = "/home/vivepat4/datasets/fire_dataset/Segmentation_Mask/Fire"
    dataset_path    = ["/home/martgro1/.kaggle/datasets/diversisai/fire-segmentation-image-dataset/versions/1/Image/Fire"]
    mask_path       = ["/home/martgro1/.kaggle/datasets/diversisai/fire-segmentation-image-dataset/versions/1/Segmentation_Mask/Fire"]

    sim_data = BinarySegmentationData(train_dir=dataset_path,
                                      mask_dir=mask_path,
                                      train_transform=train_transform)

    train_set, val_set = sim_data.split()
    print(len(train_set))
    print(len(val_set))

    the_set = train_set

    plt.figure(figsize=(30, 20))
    plt.subplots_adjust(left=0.02, right=0.98)
    indices = np.random.permutation(len(the_set))[:8]

    for i, idx in enumerate(indices):
        img, mask = the_set[idx]

        plt.subplot(4, 4, (2 * i) + 1)
        plt.imshow(img.numpy().transpose((1, 2, 0)))
        plt.colorbar()
        plt.subplot(4, 4, (2 * i + 1) + 1)
        plt.imshow(mask.numpy()[0])
        plt.colorbar()
        plt.suptitle(f"{img.shape} - {mask.shape}")
    plt.savefig("segmentation_examples.png")
