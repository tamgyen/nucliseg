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
from tqdm import tqdm

from keypoint_utils import get_keypoints_from_heatmap_batch_maxpool

PIL.Image.MAX_IMAGE_PIXELS = 933120000

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class KeypointsOnImage:
    def __init__(self,
                 image: np.ndarray = None,
                 keypoints: np.ndarray = None,
                 dir: str = None,
                 id: tuple[int, int] = None):

        self.masks = {}
        if image is not None:
            self.image = image

        if keypoints is not None:
            self.keypoints = keypoints

        if id is not None:
            self.id = id

        if dir is not None and id is not None:
            self.image = io.imread(f'{dir}/{id[0]}_{id[1]}.png')
            self.keypoints = np.load(f'{dir}/{id[0]}_{id[1]}.npy')

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)

        io.imsave(f'{dir}/{self.id[0]}_{self.id[1]}.png', self.image)
        np.save(f'{dir}/{self.id[0]}_{self.id[1]}.npy', self.keypoints)

    def add_masks(self, masks):
        self.masks = {}
        for keypoint_index in range(len(self.keypoints)):
            single_mask = masks == keypoint_index + 4
            single_mask = single_mask.astype(np.uint8)

            mean_color_hsv = self.get_mask_color(self.image, single_mask)

            x, y, w, h = cv2.boundingRect(single_mask)
            single_mask = single_mask[y:y + h, x:x + w]

            area = np.sum(single_mask)

            image_mask = self.image[y:y + h, x:x + w, :]

            text = f"H: {round(mean_color_hsv[0])}, " \
                   f"S: {round(mean_color_hsv[1])}, " \
                   f"V: {round(mean_color_hsv[2])}, " \
                   f"area: {area}"
            fig, ax = plt.subplots()
            ax.imshow(image_mask)
            ax.axis('off')
            ax.set_title(text, fontsize=12)
            plt.show()

            self.masks.update({keypoint_index: {
                'mask': single_mask,
                'coords': (x, y, w, h),
                'color': mean_color_hsv,
                'area': area
            }
            })

    def add_filter_masks(self, masks=None, draw_settings=None):

        for keypoint_index in range(len(self.keypoints)):
            x, y = self.keypoints[keypoint_index].astype(np.uint32)

            mask = masks == keypoint_index + 4

            mask = mask.astype(np.uint8)

            # check color
            color = self.get_mask_color(self.image, mask)

            sub = draw_settings.get('color_reference')-color
            dists = np.linalg.norm(sub, axis=1)

            color_class = draw_settings.get('classes')[np.argmin(dists)]

            # check area
            area = np.sum(mask)

            if draw_settings.get('max_area') > area > draw_settings.get('min_area'):

                if draw_settings.get('round_contour') > 0:
                    open_kernel = draw_settings.get('round_contour')
                    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))

                    eroded = cv2.morphologyEx(opened, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8),
                                              iterations=draw_settings.get('contour_strength'))

                    contour = opened - eroded

                else:
                    eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8),
                                              iterations=draw_settings.get('contour_strength'))
                    contour = mask - eroded

                self.image[contour > 0] = color_class

            else:

                cv2.circle(img=self.image, center=(x, y), radius=20, color=color_class, thickness=2)

            cv2.circle(img=self.image, center=(x, y), radius=2, color=color_class, thickness=2)

        return True


    @staticmethod
    def get_mask_color(image, mask):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mean = cv2.mean(image_hsv, mask=mask)

        return np.array(mean[:-1])

    def plot_masks_on_image(self, image):
        restored = np.zeros_like(image)

        bin_temp = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        bin_temp_2 = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for mask in self.masks.values():
            x, y, w, h = mask.get('coords')

            bin_temp_2[y:y + h, x:x + w] = mask.get('mask')
            bin_temp = cv2.bitwise_or(bin_temp, bin_temp_2)

        restored[:, :, 2] = bin_temp * 255

        result = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1 - .5, restored, .5, 0)

        return result

    def plot_keypoints_on_image(self, image=None, radius=4, color=None):
        if image is None:
            image = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2BGR)

        if color is None:
            color = [0, 0, 255]

        for point in self.keypoints:
            x, y = point.astype(np.uint32)
            image = cv2.circle(image, (x, y), radius, color, -1)

        return image


def split_image(image: np.ndarray, kernel_size: int) -> np.ndarray:
    h, w, c = image.shape
    im_stack = image.reshape((h // kernel_size, kernel_size, w // kernel_size, kernel_size, c))
    im_stack = im_stack.swapaxes(1, 2)

    im_stack = im_stack.reshape(-1, im_stack.shape[2], im_stack.shape[3], im_stack.shape[-1])

    return im_stack


def image_stack_to_batch_tensor(image_stack, batch_size=16):
    batches = []
    batch = []
    i = 1
    for image in tqdm(image_stack, desc='Batching'):
        tensor = to_tensor(image)
        tensor = tensor.unsqueeze(0)
        batch.append(tensor)

        if i % batch_size == 0:
            batch_t = torch.concat(batch)
            batches.append(batch_t)
            batch = []

        i += 1

    return batches


if __name__ == '__main__':

    model_input_shape = [3, 256, 256]
    DOWNSAMPLE_FROM = 1024
    BATCH_SIZE = 16

    image_src = io.imread('C:/Dev/Projects/nucliseg/01_data/src.jpg')

    im_stack = split_image(image_src, DOWNSAMPLE_FROM)

    # for im in im_stack:
    #     plt.imshow(im)
    #     plt.show()

    # ******** LOAD MODEL *******************

    model_path = "../03_models/nucleus-keypoint/6hjudtgc/epoch=29-step=6690.pt"

    model = torch.load(model_path)

    model.eval()

    model.to(device)

    # ********** INFERENCE *********************
    # images_t = to_tensor(images)

    batches = image_stack_to_batch_tensor(im_stack, batch_size=BATCH_SIZE)

    k = 0
    for batch_of_images in tqdm(batches, desc='Predicting'):
        batch_of_images = F.resize(batch_of_images, model_input_shape[1:])
        batch_of_images = batch_of_images.to(device)

        # images_t = images_t.unsqueeze(0)

        with torch.no_grad():
            heatmaps = model(batch_of_images)

        keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, max_keypoints=500, min_keypoint_pixel_distance=2)

        keypoint_predicted_images = []
        for keypoint in keypoints:
            kp = np.array(keypoint[0])
            kp_rescaled = kp * DOWNSAMPLE_FROM / model_input_shape[1]

            image = im_stack[k]
            kpoi = KeypointsOnImage(image=image, keypoints=kp_rescaled,
                                    id=(k // BATCH_SIZE, (k + BATCH_SIZE) % BATCH_SIZE))

            # kpoi_im = kpoi.plot_keypoints_on_image(image)
            # plt.imshow(kpoi_im)
            # plt.show()

            kpoi.save('../01_data/kpoi_store')

            k += 1

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
