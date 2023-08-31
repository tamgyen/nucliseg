from matplotlib import pyplot as plt

from predict import KeypointsOnImage
import cv2
import numpy as np

from skimage.segmentation import watershed
from skimage.filters import threshold_multiotsu
from skimage import feature
from skimage import filters


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




NUM_OTSU_CLASSES = 3



kpoi = KeypointsOnImage(dir='../01_data/kpoi_store', id=0)

image = kpoi.image

roi_gen = kpoi.get_roi(100)
for _ in range(1):
    coords, roi = next(roi_gen)

image_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.uint8)

otsu_th = threshold_multiotsu(image_gray, classes=NUM_OTSU_CLASSES)
markers = np.zeros_like(image_gray)

markers[image_gray > otsu_th[NUM_OTSU_CLASSES - 2]] = 255

# print(":)")
#
# cv2.imshow('im_gray', edges2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.imshow(markers)
plt.show()
