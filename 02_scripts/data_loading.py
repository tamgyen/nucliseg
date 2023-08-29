import glob

import PIL.Image
import numpy as np

import torch
from matplotlib import pyplot as plt

from torchvision import disable_beta_transforms_warning
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import datapoints as dp
from torchvision.transforms.v2 import functional as F

import torch.utils.data

from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

disable_beta_transforms_warning()


class EndoNukeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.imgs = sorted(glob.glob(f'{root}/images/*.png'))
        self.masks = sorted(glob.glob(f'{root}/masks/*.npz'))

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = read_image(img_path, mode=ImageReadMode.RGB)
        mask_np = np.load(mask_path)
        masks = torch.tensor(mask_np.get('masks'), dtype=torch.uint8)

        num_obj = masks.shape[0]
        imsize = F.get_spatial_size(img)

        boxes = masks_to_boxes(masks)

        labels = torch.ones((num_obj,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_obj,), dtype=torch.int64)

        img = dp.Image(img)

        target = {"boxes": dp.BoundingBox(boxes, format="XYXY", spatial_size=(imsize[0], imsize[1])),
                  "masks": dp.Mask(masks),
                  "labels": labels,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def show(sample, show_masks=True, show_boxes=True):
    if isinstance(sample[0], tuple):
        image = sample[0][0]
        target = sample[1][0]
    else:
        image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)

    masks = target.get('masks')

    masks = masks.to('cpu')
    image = image.to('cpu')

    annotated_image = image

    if show_masks:
        for mask in masks:
            annotated_image = draw_segmentation_masks(annotated_image, mask.to(torch.bool), alpha=0.5,
                                                      colors="yellow")
    if show_boxes:
        annotated_image = draw_bounding_boxes(annotated_image, target.get("boxes"), colors="blue", width=3)

    annotated_image_np = annotated_image.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots()
    ax.imshow(annotated_image_np)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()


if __name__ == '__main__':
    ds = EndoNukeDataset(root='../01_data/01_dataset_endonuke')

    example = ds[50]

    show(example)

