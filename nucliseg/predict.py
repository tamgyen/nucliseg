import multiprocessing
from functools import partial
from time import perf_counter

import cv2
import numpy as np
from tqdm import tqdm

from nucliseg.contour import restore_contours, stitch_image
from nucliseg.keypoints import predict_keypoints


def main(source_image: str, target_image: str, **kwargs):

    t_start = perf_counter()
    images_with_keypoints = predict_keypoints(source_image, **kwargs)
    t_stop = perf_counter()

    t_predict_keypoints = t_stop-t_start

    num_cores = multiprocessing.cpu_count()

    partial_worker = partial(restore_contours, **kwargs)

    print(f'\nRestoring contours using {num_cores} CPU cores..\n')
    t_start = perf_counter()
    with multiprocessing.Pool(processes=num_cores) as pool:
        output_kpois = list(tqdm(pool.imap(partial_worker, images_with_keypoints),
                                 total=len(images_with_keypoints),
                                 desc='Contouring'))
    t_stop = perf_counter()

    t_restore_contours = t_stop - t_start

    print(f'\nStitching and writing image..\n')

    t_start = perf_counter()
    image_stitched = stitch_image(output_kpois)
    cv2.imwrite(target_image, image_stitched)
    t_stop = perf_counter()

    t_stitching_writing = t_stop - t_start

    print(f'\nDONE!\n'
          f'Total time: {t_predict_keypoints+t_restore_contours+t_stitching_writing:.2f}s of which\n'
          f'\tt_predict_keypoints: {t_predict_keypoints:.2f}s\n'
          f'\tt_restore_contours: {t_restore_contours:.2f}s\n'
          f'\tt_stitching_writing: {t_stitching_writing:.2f}s\n')


if __name__ == '__main__':
    main('../01_data/src.jpg', target_image='../01_data/dest.jpg')

    # images_with_keypoints = predict_keypoints('../01_data/src.jpg')
    # for kpoi in images_with_keypoints:
    #     drawn_stack = restore_contours(kpoi)
