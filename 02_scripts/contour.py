import multiprocessing

from matplotlib import pyplot as plt

from predict import KeypointsOnImage
import cv2
import numpy as np

from skimage.segmentation import watershed
from skimage.filters import threshold_multiotsu
from skimage import feature
from skimage import filters
from skimage.segmentation import clear_border

from skimage import color

from tqdm import tqdm
from itertools import permutations


def place_seeds(keypoints, imsize, square_size):
    canvas = np.zeros((imsize[0], imsize[1]), dtype=np.uint8)
    for label, point in enumerate(keypoints):
        x, y = point.astype(np.uint32)

        canvas[max([y - square_size // 2, 0]):min(y + square_size // 2, imsize[0] - 1),
        max([x - square_size // 2, 0]):min([x + square_size // 2, imsize[1] - 1])] = 3 + label

    return canvas


def filter_masks(kpoi, max_area, min_area, open_kernel=5):
    kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return opening


def color_adjust(image, blue_scaling_factor=2.0, saturation_scaling_factor=1, contrast_scaling_factor=1.1):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hsv_image = image_hsv.astype(np.float32) / 255.0

    hsv_image[..., 1] *= blue_scaling_factor

    # Modify the S-channel (saturation) and V-channel (value)
    hsv_image[..., 1] *= saturation_scaling_factor  # Increase saturation
    hsv_image[..., 2] *= contrast_scaling_factor  # Increase contrast

    # Ensure that the S-channel and V-channel values are in the valid range [0, 1]
    hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 1)
    hsv_image[..., 2] = np.clip(hsv_image[..., 2], 0, 1)

    # Convert the modified HSV image back to the uint8 format
    hsv_image = (hsv_image * 255.0).astype(np.uint8)

    # Convert the HSV image back to BGR format
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return bgr_image


def extract_background(bgr_image, dilate_kernel=np.ones((3, 3), np.uint8), dilate_iterations=10):
    image_gray = bgr_image[:, :, 2]

    ret_1, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)

    sure_bg = cv2.dilate(thresh, kernel, iterations=dilate_iterations)

    return sure_bg


def watershed_from_points(kpoi, seed_size):
    image = kpoi.image

    bgr_image = color_adjust(image)

    sure_bg = extract_background(bgr_image, dilate_iterations=10)

    markers = place_seeds(kpoi.keypoints, (sure_bg.shape[0], sure_bg.shape[1]), seed_size)

    sure_fg = np.zeros_like(markers)
    sure_fg[markers > 2] = 255

    ambiguous = cv2.subtract(sure_bg, sure_fg)

    markers += 1

    markers[ambiguous == 255] = 0

    markers = cv2.watershed(image=bgr_image, markers=markers.astype(np.int32))

    return markers


def plot_contours_and_keypoints(kpoi, markers):
    image = kpoi.image

    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)
    image_contour = image_cv
    image_contour[markers == -1] = [255, 0, 255]

    image_contour_kp = kpoi.plot_keypoints_on_image(image_contour)

    #
    # image_masks = color.label2rgb(markers, bg_label=0)

    # image_assemble = np.zeros((image_bundaries.shape[0], image_bundaries.shape[1] * 2, 3))
    #
    # image_assemble[:, :image_bundaries.shape[1], :] = image_bundaries
    # image_assemble[:, image_bundaries.shape[1]:, :] = image_masks

    return image_contour_kp

    # cv2.imshow('Image Assemble', image_bundaries)
    # cv2.imshow('im_bounds', image_masks)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def restore_contours(id):
    kpoi = KeypointsOnImage(dir='../01_data/kpoi_store', id=id)
    markers = watershed_from_points(kpoi, seed_size=14)

    kpoi.add_masks(markers)

    # image_with_contours = plot_contours_and_keypoints(kpoi, markers)

    return kpoi


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


areas = []
colors = []
#
for i in tqdm(range(16)):
    for j in range(16):
        kpoi = KeypointsOnImage(dir='../01_data/kpoi_store', id=(i, j))

        markers = watershed_from_points(kpoi, seed_size=14)

        # image_with_contours = plot_contours_and_keypoints(kpoi, markers)

        # -------------------- POSTPROC -------------------------------
        draw_settings = {'min_area': 50,
                         'max_area': 5000,
                         'color_ranges': {'red': (112, 210, 71),
                                          'blue': (10, 53, 198),
                                          'orange': (109, 87, 125),
                                          'yellow': (40, 26, 200)},
                         'round_contour': 5,
                         'contour_strength': 3
                         }

        kpoi.add_filter_masks(markers, draw_settings)

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
