import cv2
import numpy as np
import os
import errno
from os import path

import weightMap as wm
from blending import (gaussPyramid, laplPyramid, blend, collapse, viz_pyramid)

SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"
EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])
MIN_DEPTH = 4

def naive(img_stack, weight_map):
    """This function implements the naive method described in "Exposure Fusion" by T. Mertens et al.
    It simply blends the input images according to their calculated weight maps which are calculated
    based on each pixel contrast, saturation and well-exposedness criteria described in the paper.
    """

    no_images = len(img_stack)
    naive_out = np.zeros((len(img_stack[0]), len(img_stack[0][0]), len(img_stack[0][0][0])), dtype=np.float64)

    for i in range(len(img_stack[0])):
        for j in range(len(img_stack[0][0])):
            for f in range(no_images):
                naive_out[i][j][:] += img_stack[f][i][j][:] * weight_map[f][i][j]

            temp = 0

    return naive_out


def multiresolution(img_stack, weight_map):
    """This function implements the multiresolution method described in "Exposure Fusion" by T. Mertens et al.
    It uses the laplacian and gaussian pyramids for blending the input images according to their weight maps
    (instead of the mask) to produces seamless final artifacts.
    """

    no_images = len(img_stack)
    img_list = []
    weight_list = []
    new_img_list = []
    new_weight_list = []
    outpyr = []

    for f in range(no_images):

        img = np.atleast_3d(img_stack[f]).astype(np.float)
        weight = np.atleast_3d(weight_map[f]).astype(np.float)
        shape = weight.shape[1::-1]
        min_size = min(img.shape[:2])
        depth = int(np.log2(min_size)) - MIN_DEPTH

        gauss_pyr_weight = [gaussPyramid(ch, depth) for ch in np.rollaxis(weight, -1)]
        gauss_pyr = [gaussPyramid(ch, depth) for ch in np.rollaxis(img, -1)]
        lapl_pyr = [laplPyramid(ch) for ch in gauss_pyr]

        img_list.append(lapl_pyr)
        weight_list.append(gauss_pyr_weight)

        viz_pyramid(gauss_pyr_weight, shape, path.join(OUT_FOLDER, 'gauss_pyr_weight' + str(f)), norm=True)
        viz_pyramid(lapl_pyr, shape, path.join(OUT_FOLDER, 'lapl_pyr_img' + str(f)), norm=True)

    """ This 'for' changes (no_images X no_channels X no_layers) order to (no_channels X no_images X no_layers) 
    so that the already implemented blend function can be used for all the three channels."""
    for ch in range(3):

        new_img_list.append([])
        new_weight_list.append([])

        for img in range(no_images):

            new_img_list[ch].append(img_list[img][ch][:])
            new_weight_list[ch].append(weight_list[img][0][:])

        outpyr.append(blend(new_img_list[ch], new_weight_list[ch]))

    outimg = [[collapse(x)] for x in outpyr]

    viz_pyramid(outpyr, shape, path.join(OUT_FOLDER, 'outpyr'), norm=True)
    viz_pyramid(outimg, shape, path.join(OUT_FOLDER, 'outimg'))


def main(image_files, output_folder):

    img_stack = [cv2.imread(name) for name in image_files
                 if path.splitext(name)[-1][1:].lower() in EXTENSIONS]

    weight_map = wm.computeWeightMap(img_stack)
    out_naive = naive(img_stack, weight_map)
    multiresolution(img_stack, weight_map)
    cv2.imwrite(path.join(OUT_FOLDER, "outimg_naive.png"), out_naive)

    print("Done!")


if __name__ == "__main__":

    np.random.seed()
    src_contents = os.walk(SRC_FOLDER)
    for dirpath, _, fnames in os.walk(SRC_FOLDER):
        image_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(OUT_FOLDER, image_dir)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    image_files = sorted([os.path.join(dirpath, name) for name in fnames])

    main(image_files, output_dir)
