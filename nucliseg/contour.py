from nucliseg.keypoints import KeypointsOnImage

import cv2
import numpy as np
from tqdm import tqdm


def place_seeds(keypoints: tuple, imsize: tuple, radius: int):
    """
    Draws a circle of scalar labelled pixels around keypoints onto a canvas.
    """

    canvas = np.zeros((imsize[0], imsize[1], 3), dtype=np.uint8)
    for label, point in enumerate(keypoints):
        x, y = point.astype(np.uint32)

        canvas = cv2.circle(canvas, (x, y), radius, (label + 3, label + 3, label + 3), thickness=-1)

    return canvas[..., 0]


def color_adjust(image: np.ndarray,
                 blue_scaling_factor: float,
                 saturation_scaling_factor: float,
                 contrast_scaling_factor: float) -> np.ndarray:
    """
    Simple color adjustments for easier background separation and to help increase gradients for watershed where needed.
    """

    # convert to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # normalize
    hsv_image = image_hsv.astype(np.float32) / 255.0

    # scale
    hsv_image[..., 1] *= blue_scaling_factor
    hsv_image[..., 1] *= saturation_scaling_factor
    hsv_image[..., 2] *= contrast_scaling_factor

    # clip between [0, 1]
    hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 1)
    hsv_image[..., 2] = np.clip(hsv_image[..., 2], 0, 1)

    # rescale
    hsv_image = (hsv_image * 255.0).astype(np.uint8)

    # back to BGR
    image_adjusted = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return image_adjusted


def extract_background(image: np.ndarray, dilate_iter_background: int) -> np.ndarray:
    """
    This function extracts the pixels that are certainly background.
    Parameters
    """

    # get blue channel
    image_gray = image[..., 2]

    # adaptive thresholding using Otsu based on histogram
    ret_1, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # swap fg and bg
    thresh = cv2.bitwise_not(thresh)

    # remove noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # increase fg area adjacent to already fg pixels to be sure not to exclude any objects
    sure_bg = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=dilate_iter_background)

    # visu
    # overlay = np.zeros_like(bgr_image)
    #
    # overlay[sure_bg > 0] = [255, 255, 255]
    #
    # result = cv2.addWeighted(bgr_image, 1 - .5, overlay, .5, 0)
    #
    # cv2.imshow("sure_bg", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return sure_bg


def watershed_from_points(kpoi: KeypointsOnImage,
                          seed_radius: int,
                          dilate_iter_background: int,
                          blue_scaling_factor: float,
                          saturation_scaling_factor: float,
                          contrast_scaling_factor: float):
    """
    Preprocesses the image for watershed segmentation using some color adjustment.Also extracts background, foreground,
    and places seeds around keypoints. Finally executes watershed.
    reference: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    """

    # get the image
    image = kpoi.image

    # adjust color
    adjusted_image = color_adjust(image, blue_scaling_factor, saturation_scaling_factor, contrast_scaling_factor)

    # extract bg
    sure_bg = extract_background(adjusted_image, dilate_iter_background=dilate_iter_background)

    # get markers (scalar labelled mask) for watershed: use predicted keypoints for seeds
    markers = place_seeds(kpoi.keypoints, (sure_bg.shape[0], sure_bg.shape[1]), seed_radius)

    # get foreground (just the binary mask of markers)
    sure_fg = np.zeros_like(markers)
    sure_fg[markers > 2] = 255

    # get possible areas to also label as objects
    ambiguous = cv2.subtract(sure_bg, sure_fg)

    # change bg label to 1
    markers += 1

    # mark unknown with 0
    markers[ambiguous == 255] = 0

    # execute segmentation
    markers = cv2.watershed(image=adjusted_image, markers=markers.astype(np.int32))

    return markers


def restore_contours(kpoi: KeypointsOnImage, **kwargs) -> KeypointsOnImage:
    """
    This function restores the contours of nuclei around keypoints on the image using watershed algo and then also
    draws the masks on the image.

    """

    # store some default settings.. should be moved to a config file
    default_draw_settings = {'min_area': 50,
                             'max_area': 6000,
                             'color_reference': np.array([[8, 160, 110],  # red
                                                          [10, 190, 70],  # red
                                                          [13, 152, 149],  # red
                                                          [114, 60, 198],  # blue
                                                          [15, 87, 125],  # orange
                                                          [115, 26, 200]]),  # yellow
                             'classes': [(255, 0, 0),
                                         (255, 0, 0),
                                         (255, 0, 0),
                                         (0, 0, 255),
                                         (255, 163, 0),
                                         (255, 255, 0)],
                             'round_contour': 5,
                             'contour_strength': 2
                             }

    # parse kwargs
    seed_radius = kwargs.pop('seed_radius', 12)
    dilate_iter_background = kwargs.pop('dilate_iter_background', 10)
    blue_scaling_factor = kwargs.pop('blue_scaling_factor', 2)
    saturation_scaling_factor = kwargs.pop('saturation_scaling_factor', 1.2)
    contrast_scaling_factor = kwargs.pop('contrast_scaling_factor', 1.1)

    draw_settings = kwargs.pop('draw_settings', default_draw_settings)

    # watershed to restore contours
    scalar_labelled_masks = watershed_from_points(kpoi,
                                                  seed_radius=seed_radius,
                                                  dilate_iter_background=dilate_iter_background,
                                                  blue_scaling_factor=blue_scaling_factor,
                                                  saturation_scaling_factor=saturation_scaling_factor,
                                                  contrast_scaling_factor=contrast_scaling_factor)

    # filter and draw valid contours
    kpoi.filter_draw_masks(masks=scalar_labelled_masks,
                           draw_settings=draw_settings)

    return kpoi


def stitch_image(kpois: list[KeypointsOnImage]):
    """
    This function can reassemble the source image based on the objects' ID property.
    """

    # calculate tile size
    tile_size = kpois[0].image.shape[0]

    # and original image dims
    original_image_size = (np.sqrt(len(kpois)) * tile_size).astype(np.int32)

    # preallocate
    canvas = np.zeros((original_image_size, original_image_size, 3))

    # draw kpoi images based on their position ID
    for kpoi in tqdm(kpois, desc='Stitching'):
        x = kpoi.id[1] * tile_size
        y = kpoi.id[0] * tile_size

        canvas[y:y + tile_size, x:x + tile_size, :] = cv2.cvtColor(kpoi.image, cv2.COLOR_RGB2BGR)

    return canvas
