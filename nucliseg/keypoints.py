import os
import PIL.Image
import cv2

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as F

from skimage import io
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 933120000  # handle large images


class KeypointsOnImage:
    """
    Class for conveniently storing image tiles and their keypoints together. Also contains some helper functions related
    to them.
    """

    def __init__(self,
                 image: np.ndarray = None,
                 keypoints: np.ndarray = None,
                 dir: str = None,
                 id: tuple[int, int] = None):
        """
        We can construct object from image+keypoint objets or from saved data.
        """

        if image is not None:
            self.image = image

        if keypoints is not None:
            self.keypoints = keypoints

        if id is not None:
            self.id = id

        if dir is not None and id is not None:
            self.image = io.imread(f'{dir}/{id[0]}_{id[1]}.png')
            self.keypoints = np.load(f'{dir}/{id[0]}_{id[1]}.npy')

    def save(self, dir: str):
        """
        Save objects as png and npy files to dir.
        """

        os.makedirs(dir, exist_ok=True)

        io.imsave(f'{dir}/{self.id[0]}_{self.id[1]}.png', self.image)
        np.save(f'{dir}/{self.id[0]}_{self.id[1]}.npy', self.keypoints)

    # TODO: this should be split into separate methods..
    def filter_draw_masks(self, masks: np.ndarray, draw_settings: dict):
        """
        This function iterates over the keypoints once, assigns masks from watershed, filters them based on size, then
        dilates them to remove misshapen areas and finally restores the contours and draws them.
        """

        color_reference = draw_settings.get('color_reference')
        classes = draw_settings.get('classes')

        # iterate through keypoints
        for i in range(len(self.keypoints)):
            x, y = self.keypoints[i].astype(np.uint32)

            # get mask from watershed result based on label
            mask = masks == i + 4
            mask = mask.astype(np.uint8)

            # classify based on color
            color_class = self.get_color_class(mask,
                                               color_reference,
                                               classes)

            # if valid based on area draw the contour, if not draw a circle
            if draw_settings.get('max_area') > np.sum(mask) > draw_settings.get('min_area'):

                # remove misshapen parts by opening.. this also makes the masks more circular
                if draw_settings.get('round_contour') > 0:
                    open_kernel = draw_settings.get('round_contour')
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))

                # get the contour.. select size by number of erode iters
                eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8),
                                          iterations=draw_settings.get('contour_strength'))
                contour = mask - eroded

                # draw the contour
                self.image[contour > 0] = color_class

            else:

                cv2.circle(img=self.image, center=(x, y), radius=20, color=color_class, thickness=2)

            # draw keypoints
            cv2.circle(img=self.image, center=(x, y), radius=2, color=color_class, thickness=2)

    @staticmethod
    def get_mask_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Calculates mean color of the image below a mask.
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mean = cv2.mean(image_hsv, mask=mask)

        return np.array(mean[:-1])

    def get_color_class(self, mask: np.ndarray,
                        color_reference: np.ndarray,
                        classes: list[tuple]) -> tuple:
        """
        Classify a mask based on the L2 distance of its mean color from some reference colors.
        """
        color = self.get_mask_color(self.image, mask)

        sub = color_reference - color
        dists = np.linalg.norm(sub, axis=1)

        return classes[np.argmin(dists)]


def get_keypoints_from_heatmap_batch_maxpool(
        heatmap: torch.Tensor,
        max_keypoints: int = 500,
        min_keypoint_pixel_distance: int = 3,
        abs_max_threshold: float = None,
        rel_max_threshold: float = None,
        return_scores: bool = False,
):
    """
    reference: https://github.com/tlpss/keypoint-detection/tree/main

    Fast extraction of keypoints from a batch of heatmaps using maxpooling.

    Inspired by mmdetection and CenterNet:
      https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html
    """

    batch_size, n_channels, _, width = heatmap.shape

    # obtain max_keypoints local maxima for each channel (w/ maxpool)
    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = min_keypoint_pixel_distance

    # exclude border keypoints by padding with highest possible value
    # bc the borders are more susceptible to noise and could result in false positives
    padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
    max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)

    # if the value equals the original value, it is the local maximum
    local_maxima = max_pooled_heatmap == heatmap

    # all values to zero that are not local maxima
    heatmap = heatmap * local_maxima

    # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
    scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
    indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)

    # moving them to CPU now to avoid multiple GPU-mem accesses!
    indices = indices.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
    filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

    # determine NMS threshold
    threshold = 0.01  # make sure it is > 0 to filter out top-k that are not local maxima
    if abs_max_threshold is not None:
        threshold = max(threshold, abs_max_threshold)
    if rel_max_threshold is not None:
        threshold = max(threshold, rel_max_threshold * heatmap.max())

    # have to do this manually as the number of maxima for each channel can be different
    for batch_idx in range(batch_size):
        for channel_idx in range(n_channels):
            candidates = indices[batch_idx, channel_idx]
            for candidate_idx in range(candidates.shape[0]):

                # these are filtered out directly.
                if scores[batch_idx, channel_idx, candidate_idx] > threshold:
                    # convert to (u,v)
                    filtered_indices[batch_idx][channel_idx].append(candidates[candidate_idx][::-1].tolist())
                    filtered_scores[batch_idx][channel_idx].append(scores[batch_idx, channel_idx, candidate_idx])
    if return_scores:
        return filtered_indices, filtered_scores
    else:
        return filtered_indices


def split_image(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Splits the image into "kernel_size" sized tiles.
    """
    h, w, c = image.shape

    im_stack = image.reshape((h // kernel_size, kernel_size, w // kernel_size, kernel_size, c))

    im_stack = im_stack.swapaxes(1, 2)

    im_stack = im_stack.reshape(-1, im_stack.shape[2], im_stack.shape[3], im_stack.shape[-1])

    return im_stack


def image_stack_to_batch_tensor(image_stack: np.ndarray, batch_size: int) -> list[torch.Tensor]:
    """

    Parameters
    ----------
    image_stack
    batch_size

    Returns
    -------

    """
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


def predict_keypoints(image_path: str,
                      **kwargs) -> list[KeypointsOnImage]:
    """
    This is the main function for predicting keypoint locations on the input image.
    """

    # parse kwargs
    model_path = kwargs.pop('model_path', os.path.dirname(__file__) + '/models/keypoint_detector_6hjudtgc.pt')
    model_input_shape = kwargs.pop('model_input_shape', [3, 256, 256])
    tile_size = kwargs.pop('tile_size', 1024)
    batch_size = kwargs.pop('batch_size', 16)
    min_keypoint_pixel_distance = kwargs.pop('min_keypoint_pixel_distance', 2)
    write_to_file = kwargs.pop('write_to_file', False)
    save_dir = kwargs.pop('save_dir', '../01_data/kpoi_store')

    # check hw
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model and transfer weight to gpu
    model = torch.jit.load(model_path)

    model.eval()
    model.to(device)

    # reading with pil.. just to ensure model compatibility
    image_src = io.imread(image_path)

    # split image to tiles for model
    im_stack = split_image(image_src, tile_size)

    tile_dim = np.sqrt(len(im_stack))

    # make a list of batches
    batches = image_stack_to_batch_tensor(im_stack, batch_size=batch_size)

    # main loop for pred: this would be the place for multi-gpu preds..
    b = 0  # b for batch
    keypoint_annotated_images = []
    for batch_of_images in tqdm(batches, desc='Predicting'):

        # we can only batch resize on cpu, but that's ok
        batch_of_images = F.resize(batch_of_images, model_input_shape[1:])
        batch_of_images = batch_of_images.to(device)

        # forward prop
        with torch.no_grad():
            heatmaps = model(batch_of_images)

        # convert our heatmap batch to a list of keypoints using helper from the keypoint repo
        batch_of_keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps,
                                                                      max_keypoints=500,
                                                                      min_keypoint_pixel_distance=min_keypoint_pixel_distance)

        for keypoints in batch_of_keypoints:

            # rescale keypoints back to original tile reference
            keypoints_rescaled = np.array(keypoints[0]) * tile_size / model_input_shape[1]

            # store images together with their keypoints for tidiness
            image = im_stack[b]
            kpoi = KeypointsOnImage(image=image, keypoints=keypoints_rescaled,
                                    id=(int(b // tile_dim), int((b + tile_dim) % tile_dim)))

            keypoint_annotated_images.append(kpoi)

            # write to files if needed
            if write_to_file:
                os.makedirs(save_dir)
                kpoi.save(save_dir)

            b += 1

    return keypoint_annotated_images


if __name__ == '__main__':
    kpois = predict_keypoints('../01_data/src.jpg', )
