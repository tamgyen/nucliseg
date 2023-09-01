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


def compute_image_grad(image):
    """
    Compute image gradient. Each pixel stores the maximum from x- and y- gradients.

    Parameters
    ----------
    image: nd.array
        image to take the gradient from. If it has multiple channales,
        the gradyscale version is taken before computeng the gradient.

    Returns
    -------
    image_grad_abs: ndarray
        image gradient (maxim values from x- and x-y gradients at a given point).

    """
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    else:
        image_gray = image.astype(np.uint8)

    image_grad_x = np.abs(cv2.Sobel(image_gray, cv2.CV_64F, 1, 0))
    image_grad_y = np.abs(cv2.Sobel(image_gray, cv2.CV_64F, 0, 1))
    image_grad_x = image_grad_x * 255 / image_grad_x.max()
    image_grad_y = image_grad_y * 255 / image_grad_y.max()
    image_grad_abs = np.where(
        image_grad_x > image_grad_y, image_grad_x, image_grad_y
    ).astype(np.uint8)
    return image_grad_abs


def place_seeds(keypoints, imsize, square_size):
    canvas = np.zeros((imsize[0], imsize[1]), dtype=np.uint8)
    for label, point in enumerate(keypoints):
        x, y = point.astype(np.uint32)

        canvas[max([y - square_size // 2, 0]):min(y + square_size // 2, imsize[0] - 1),
        max([x - square_size // 2, 0]):min([x + square_size // 2, imsize[1] - 1])] = 3 + label

    return canvas


NUM_OTSU_CLASSES = 3
ROI_SIZE = 100
DILATE_ITER = 15

for id in range(8):
    kpoi = KeypointsOnImage(dir='../01_data/kpoi_store', id=id)

    image = kpoi.image

    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)

    roi_gen = kpoi.get_roi(ROI_SIZE)

    # image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    image_blue = image[:, :, 0]

    image_gray = image_blue.astype(np.uint8)

    image_gray_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)

    image_to_shed = cv2.cvtColor(image_gray_blur, cv2.COLOR_GRAY2BGR)

    ret_1, thresh = cv2.threshold(image_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)

    sure_bg = cv2.dilate(thresh, kernel, iterations=DILATE_ITER)

    markers = place_seeds(kpoi.keypoints, (sure_bg.shape[0], sure_bg.shape[1]), 8)

    sure_fg = np.zeros_like(markers)
    sure_fg[markers > 2] = 255

    # ret, cc = cv2.connectedComponents(sure_fg)

    ambiguous = cv2.subtract(sure_bg, sure_fg)

    markers += 1

    markers[ambiguous == 255] = 0

    # plt.imshow(markers, cmap='jet')
    # plt.show()

    markers = markers.astype(np.int32)

    markers = cv2.watershed(image=image_to_shed, markers=markers)
    # ---------------------- VISU -----------------------------------

    image_bundaries = image_cv
    image_bundaries[markers == -1] = [255, 0, 255]

    image_masks = color.label2rgb(markers, bg_label=0)

    image_assemble = np.zeros((image_bundaries.shape[0], image_bundaries.shape[1] * 2, 3))

    image_assemble[:, :image_bundaries.shape[1], :] = image_bundaries
    image_assemble[:, image_bundaries.shape[1]:, :] = image_masks

    # cv2.imshow('Image Assemble', image_masks)
    # cv2.imshow('im_bounds', image_bundaries)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Create an all-zero array with the same shape as the RGB image
    # overlay = np.zeros_like(image)

    # Set the pixels where the binary image is nonzero to white
    # overlay[ambiguous > 0] = [255, 255, 255]
    #
    # result = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1 - .5, overlay, .5, 0)

    # cv2.imshow("sure_bg", image_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kpoi_result = KeypointsOnImage(image=cv2.cvtColor(image_bundaries, cv2.COLOR_BGR2RGB), keypoints=kpoi.keypoints)

    kpoi_result.plot(show=True)

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
