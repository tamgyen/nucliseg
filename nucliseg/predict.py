from nucliseg.contour import restore_contours, stitch_image
from nucliseg.keypoints import predict_keypoints

import argparse
import multiprocessing
from functools import partial
from time import perf_counter
import cv2
from tqdm import tqdm


def predict(source_image: str, target_image: str, **kwargs):
    t_start = perf_counter()
    images_with_keypoints = predict_keypoints(source_image, **kwargs)
    t_stop = perf_counter()

    t_predict_keypoints = t_stop - t_start

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
          f'Total runtime: {t_predict_keypoints + t_restore_contours + t_stitching_writing:.2f}s of which:\n'
          f' - t_predict_keypoints: {t_predict_keypoints:.2f}s\n'
          f' - t_restore_contours: {t_restore_contours:.2f}s\n'
          f' - t_stitching_writing: {t_stitching_writing:.2f}s\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('source_image', type=str, help='Path to the source image')
    parser.add_argument('target_image', type=str, help='Path to the target image')

    args = parser.parse_args()

    predict(args.source_image, args.target_image)
