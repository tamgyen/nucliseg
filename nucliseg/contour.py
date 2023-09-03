from nucliseg.keypoints import KeypointsOnImage

import cv2
import numpy as np
from tqdm import tqdm


def place_seeds(keypoints: tuple, imsize: tuple, radius: int):
    """
    Draws a circle of scalar labelled pixels
    Parameters
    -------

    """
    canvas = np.zeros((imsize[0], imsize[1], 3), dtype=np.uint8)
    for label, point in enumerate(keypoints):
        x, y = point.astype(np.uint32)

        canvas = cv2.circle(canvas, (x, y), radius, (label+3, label+3, label+3), thickness=-1)

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
    ----------
    image: input image
    dilate_iter_background

    Returns
    -------

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
                          blue_scaling_factor,
                          saturation_scaling_factor,
                          contrast_scaling_factor):
    """

    """

    image = kpoi.image

    # im_to_shed = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    bgr_image = color_adjust(image, blue_scaling_factor, saturation_scaling_factor, contrast_scaling_factor)

    sure_bg = extract_background(bgr_image, dilate_iter_background=dilate_iter_background)

    markers = place_seeds(kpoi.keypoints, (sure_bg.shape[0], sure_bg.shape[1]), seed_radius)

    sure_fg = np.zeros_like(markers)
    sure_fg[markers > 2] = 255

    ambiguous = cv2.subtract(sure_bg, sure_fg)

    markers += 1

    markers[ambiguous == 255] = 0

    markers = cv2.watershed(image=bgr_image, markers=markers.astype(np.int32))

    return markers


def restore_contours(kpoi: KeypointsOnImage, **kwargs) -> KeypointsOnImage:
    """

    Parameters
    ----------
    kpoi

    Returns
    -------

    """

    default_draw_settings = {'min_area': 50,
                             'max_area': 6000,
                             'color_reference': np.array([[8, 160, 110],  # red
                                                          [10, 190, 70],  # red
                                                          [13, 152, 149],  # red
                                                          [114, 60, 198],  # blue
                                                          [15, 87, 125],  # orange
                                                          [115, 26, 200]]),  # yellow
                             'classes': ((255, 0, 0),
                                         (255, 0, 0),
                                         (255, 0, 0),
                                         (0, 0, 255),
                                         (255, 163, 0),
                                         (255, 255, 0)),
                             'round_contour': 5,
                             'contour_strength': 2
                             }

    seed_radius = kwargs.pop('seed_radius', 12)
    dilate_iter_background = kwargs.pop('dilate_iter_background', 10)
    blue_scaling_factor = kwargs.pop('blue_scaling_factor', 2)
    saturation_scaling_factor = kwargs.pop('saturation_scaling_factor', 1.2)
    contrast_scaling_factor = kwargs.pop('contrast_scaling_factor', 1.1)

    draw_settings = kwargs.pop('draw_settings', default_draw_settings)

    markers = watershed_from_points(kpoi,
                                    seed_radius=seed_radius,
                                    dilate_iter_background=dilate_iter_background,
                                    blue_scaling_factor=blue_scaling_factor,
                                    saturation_scaling_factor=saturation_scaling_factor,
                                    contrast_scaling_factor=contrast_scaling_factor)

    kpoi.filter_draw_masks(masks=markers,
                           draw_settings=draw_settings)

    return kpoi


def stitch_image(kpois: list[KeypointsOnImage]):
    tile_size = kpois[0].image.shape[0]
    original_image_size = (np.sqrt(len(kpois))*tile_size).astype(np.int32)
    canvas = np.zeros((original_image_size, original_image_size, 3))

    for kpoi in tqdm(kpois, desc='Stitching'):
        x = kpoi.id[1]*tile_size
        y = kpoi.id[0]*tile_size

        canvas[y:y+tile_size, x:x+tile_size, :] = cv2.cvtColor(kpoi.image, cv2.COLOR_RGB2BGR)

    return canvas


# if __name__ == '__main__':
#     num_cores = multiprocessing.cpu_count()
#
#     print('restoring contours..')
#     with multiprocessing.Pool(processes=num_cores) as pool:
#         indices = list(permutations(range(16), 2))
#
#         indices += [(x, x) for x in range(16)]
#
#         output_kpois = pool.map(restore_contours, indices)
#
#     areas = []
#     colors = []
#
#     for kpoi in output_kpois:
#         for mask in kpoi.masks.values():
#             areas.append(mask.get('area'))
#             colors.append(mask.get('color'))
#
#     plt.hist(areas, bins=150, color='blue', alpha=0.7, rwidth=0.9)
#     plt.title('Areas of nuclei')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#
#     # Show the histogram
#     plt.show()

#

if __name__ == '__main__':
    kwargs = {
        ''
    }

    kpois = []
    for i in tqdm(range(16)):
        for j in range(16):
            kpoi = KeypointsOnImage(dir='../01_data/kpoi_store', id=(i, j))
            restore_contours(kpoi)

            cv2.imshow('drawn', cv2.cvtColor(kpoi.image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    #
    #
    # plt.hist(areas, bins=100, color='blue', alpha=0.7, rwidth=0.9)
    # plt.title('Areas of nuclei')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    #
    # # Show the histogram
    # plt.show()

    # # # ---------------------- VISU -----------------------------------
    # #
    # plotted_masks = kpoi.plot_masks_on_image(kpoi.image)
    #
    # cv2.imwrite('./first_row_unfiltered.png', output_image)
    # cv2.imshow('adjusted_masks', image_with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Create an all-zero array with the same shape as the RGB image
    # overlay = np.zeros_like(image)
    #
    # # Set the pixels where the binary image is nonzero to white
    # overlay[sure_bg > 0] = [255, 255, 255]
    #
    # result = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1 - .5, overlay, .5, 0)

    # cv2.imshow("sure_bg", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # #
    # kpoi_result = KeypointsOnImage(image=cv2.cvtColor(plotted_masks, cv2.COLOR_BGR2RGB), keypoints=kpoi.keypoints)
    #
    # kpoi_result.plot(show=True)

    # for _ in tqdm(range(50)):
    #     coords, roi = next(roi_gen)
    #
    #     image_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    #
    #     otsu_th = threshold_multiotsu(image_gray, classes=NUM_OTSU_CLASSES)
    #     markers = np.zeros_like(image_gray)
    #
    #     markers[image_gray > otsu_th[NUM_OTSU_CLASSES - 2]] = 1
    #
    #     markers[ROI_SIZE//2-2:ROI_SIZE//2+2, ROI_SIZE//2-2:ROI_SIZE//2+2] = 5
    #
    #     image_grad = compute_image_grad(image_gray)
    #
    #     mask = watershed(image_grad, markers)

    # print(":)")
    #
    # cv2.imshow('im_gray', edges2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # plt.imshow(mask)
    # plt.show()

    # cv2.imshow('im_gray', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
