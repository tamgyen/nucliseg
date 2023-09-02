import os
import typing

import PIL.Image
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as F

from skimage import io
from keypoint_utils import get_keypoints_from_heatmap_batch_maxpool

PIL.Image.MAX_IMAGE_PIXELS = 933120000

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class KeypointsOnImage:
    def __init__(self,
                 image: np.ndarray = None,
                 keypoints: np.ndarray = None,
                 dir: str = None,
                 id: typing.Union[int, str] = None):

        self.masks = None
        if image is not None:
            self.image = image
        if keypoints is not None:
            self.keypoints = keypoints
        if dir is not None and id is not None:
            self.image = io.imread(f'{dir}/{id}.png')
            self.keypoints = np.load(f'{dir}/{id}.npy')

    def save(self, dir, id):

        os.makedirs(dir, exist_ok=True)
        io.imsave(f'{dir}/{id}.png', self.image)

        np.save(f'{dir}/{id}.npy', self.keypoints)

    def plot(self, show: bool = False):
        fig, ax = plt.subplots()

        ax.imshow(self.image)

        for point in self.keypoints:
            x, y = point
            circle = plt.Circle((x, y), radius=3, color='r', fill=False)
            ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlim(0, self.image.shape[1])
        ax.set_ylim(0, self.image.shape[0])

        if show:
            plt.show()

        return ax

    # TODO: what if we are on the edge of the im?
    def get_roi(self, size):
        for keypoint in self.keypoints.astype('uint32'):
            x_0 = keypoint[1] - size // 2
            x_1 = keypoint[1] + size // 2
            y_0 = keypoint[0] - size // 2
            y_1 = keypoint[0] + size // 2

            yield (y_0, y_1, x_0, x_1), self.image[x_0:x_1, y_0:y_1, :]

    def add_masks(self, masks):
        self.masks = []
        for keypoint_index in range(len(self.keypoints)):
            single_mask = masks == keypoint_index + 4
            single_mask = single_mask.astype(np.uint8)

            self.get_mask_color(self.image, single_mask)

            x, y, w, h = cv2.boundingRect(single_mask)
            single_mask = single_mask[y:y + h, x:x + w]

            self.masks.append({'mask': single_mask,
                               'coords': (x, y, w, h),
                               })

    @staticmethod
    def get_mask_color(image, mask):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean = cv2.mean(image_hsv, mask=mask)

        return mean

    def plot_masks_on_image(self, image):
        restored = np.zeros_like(image)

        bin_temp = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        bin_temp_2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for mask in self.masks:
            x, y, w, h = mask.get('coords')

            bin_temp_2[y:y + h, x:x + w] = mask.get('mask')
            bin_temp = cv2.bitwise_or(bin_temp, bin_temp_2)

        restored[:, :, 2] = bin_temp * 255

        result = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1 - .5, restored, .5, 0)

        return result

    def plot_keypoints_on_image(self, image, radius=4, color=None):
        if color is None:
            color = [0, 0, 255]

        for point in self.keypoints:
            x, y = point.astype(np.uint32)
            cv2.circle(image, (x, y), radius, color, -1)

        return image


def split_image(image: np.ndarray, kernel_size: int) -> np.ndarray:
    h, w, c = image.shape
    im_stack = image.reshape((h // kernel_size, kernel_size, w // kernel_size, kernel_size, c))

    return im_stack.swapaxes(1, 2)


def image_stack_to_batch_tensor(image_stack):
    batch = []
    for image in image_stack:
        tensor = to_tensor(image)
        tensor = tensor.unsqueeze(0)
        batch.append(tensor)

    batch_t = torch.concat(batch)

    return batch_t


if __name__ == '__main__':

    model_input_shape = [3, 256, 256]
    DOWNSAMPLE_FROM = 1024

    image_src = io.imread('C:/Dev/Projects/nucliseg/01_data/src.jpg')

    im_stack = split_image(image_src, DOWNSAMPLE_FROM)

    images = im_stack[8:16, 0, :, :, :]

    # ******** LOAD MODEL *******************

    model_path = "../03_models/nucleus-keypoint/6hjudtgc/epoch=29-step=6690.pt"

    model = torch.load(model_path)

    model.eval()

    model.to(device)

    # ********** INFERENCE *********************
    # images_t = to_tensor(images)

    images_t = image_stack_to_batch_tensor(images)

    images_t = F.resize(images_t, model_input_shape[1:])

    images_t = images_t.to(device)

    # images_t = images_t.unsqueeze(0)

    with torch.no_grad():
        heatmaps = model(images_t)

    keypoints = \
        get_keypoints_from_heatmap_batch_maxpool(heatmaps, max_keypoints=500, min_keypoint_pixel_distance=2)

    keypoint_predicted_images = []
    for image, keypoint in zip(images, keypoints):
        kp = np.array(keypoint[0])
        kp_rescaled = kp * DOWNSAMPLE_FROM / model_input_shape[1]

        keypoint_predicted_images.append(KeypointsOnImage(image, kp_rescaled))

    for i, annotated in enumerate(keypoint_predicted_images):
        annotated.plot(show=True)
        annotated.save('../01_data/kpoi_store', i)

    # keypoints = np.array(keypoints)
    # keypoints = keypoints.squeeze()

    # ************* POSTPROC ***********************

    # keypoints_rescaled = keypoints * DOWNSAMPLE_FROM / model_input_shape[1]
    #
    # fig, ax = plt.subplots()
    #
    # # Display the RGB image
    # ax.imshow(first_image)
    #
    # # Plot circles for each keypoint
    # for point in keypoints_rescaled:
    #     x, y = point
    #     circle = plt.Circle((x, y), radius=3, color='r', fill=False)
    #     ax.add_patch(circle)
    #
    # ax.set_aspect('equal')
    # ax.invert_yaxis()  # Invert y-axis to match image coordinates
    # ax.set_xlim(0, first_image.shape[1])  # Set x-axis limits
    # ax.set_ylim(0, first_image.shape[0])  # Set y-axis limits
    #
    # plt.show()
