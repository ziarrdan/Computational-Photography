import cv2
import numpy as np
import math


def laplacian(imgs):
    """This function calculates the first quality measure for exposure fusion, namely contrast or C
    """

    # 3x3 laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    N, W = laplacian_kernel.shape
    no_images = len(imgs)
    laplacian_images = np.ndarray((len(imgs), len(imgs[0]), len(imgs[0][0])), dtype=np.float64)
    filtered_grayscale = np.ndarray((len(imgs[0]), len(imgs[0][0])), dtype=np.float64)

    for f in range(no_images):
        grayscale = cv2.cvtColor(imgs[f], cv2.COLOR_BGR2GRAY)

        # considering the fact that the absolute value of the filter response should
        # be taken according to the paper, it's not possible to use filter2D or Laplacian
        # implementations in OpenCV and the convolution should be written completely.
        padded_img = cv2.copyMakeBorder(grayscale, (N - 1) // 2, (N - 1) // 2, (N - 1) // 2, (N - 1) // 2,
                                        borderType=cv2.BORDER_REFLECT_101)

        for i in range(len(filtered_grayscale)):
            for j in range(len(filtered_grayscale[0])):
                filtered_grayscale[i, j] = abs((laplacian_kernel * padded_img[i:i + 3, j:j + 3]).sum())

        laplacian_images[f] = filtered_grayscale

    return laplacian_images


def saturation(imgs):
    """This function calculates the second quality measure for exposure fusion, namely saturation or S
    """

    no_images = len(imgs)
    saturation_images = np.ndarray((len(imgs), len(imgs[0]), len(imgs[0][0])), dtype=np.float64)

    for f in range(no_images):
        for i in range(len(imgs[0])):
            for j in range(len(imgs[0][0])):
                green = imgs[f][i][j][0] / 255.
                blue = imgs[f][i][j][1] / 255.
                red = imgs[f][i][j][2] / 255.

                mean = (green + blue + red) / 3.

                saturation_images[f][i][j] = math.sqrt((((green - mean) ** 2) + ((blue - mean) ** 2)
                                              + ((red - mean) ** 2)) / 2)

    return saturation_images


def exposedness(imgs):
    """This function calculates the third quality measure for exposure fusion, namely well-exposedness or E
    """

    no_images = len(imgs)
    exposedness_images = np.ndarray((len(imgs), len(imgs[0]), len(imgs[0][0])), dtype=np.float64)

    for f in range(no_images):

        imgs[f] = np.float32(imgs[f])

        for i in range(len(imgs[0])):
            for j in range(len(imgs[0][0])):
                green = imgs[f][i][j][0] / 255.
                blue = imgs[f][i][j][1] / 255.
                red = imgs[f][i][j][2] / 255.

                gweight = math.exp(-(((green - 0.5) ** 2) / (2 * (0.2 ** 2))))
                bweight = math.exp(-(((blue - 0.5) ** 2) / (2 * (0.2 ** 2))))
                rweight = math.exp(-(((red - 0.5) ** 2) / (2 * (0.2 ** 2))))

                exposedness_images[f][i][j] = gweight * bweight * rweight

    return exposedness_images


def normalizeWP(weightMap):
    """This function normalizes the calculated weight maps for all the input images.
    """

    no_images = len(weightMap)
    totalWeight = np.zeros((len(weightMap[0]), len(weightMap[0][0])), dtype=np.float64)
    normalWeightMap = np.ndarray((len(weightMap), len(weightMap[0]), len(weightMap[0][0])), dtype=np.float64)

    for i in range(len(weightMap[0])):
        for j in range(len(weightMap[0][0])):
            for f in range(no_images):
                totalWeight[i][j] += weightMap[f][i][j]

    for f in range(no_images):
        for i in range(len(weightMap[0])):
            for j in range(len(weightMap[0][0])):
                if totalWeight[i][j] != 0:
                    normalWeightMap[f][i][j] = weightMap[f][i][j] / totalWeight[i][j]

    return normalWeightMap


def computeWeightMap(imgs):
    """This function calculates the weight maps for all the input images by simply applying the following equation:
    W = C x S x E, for each pixel of each image. As C could be zero for many pixels in all the input images it can
    cause black holes formation on the final artifact. To prevent this from happening, computeWeightMap checks the
    value of C for each pixel and if it's equal to zero, it calculates the weight only based on the other two
    components.
    """

    C = laplacian(imgs)
    S = saturation(imgs)
    E = exposedness(imgs)

    no_images = len(imgs)
    weightMap = np.ndarray((len(imgs), len(imgs[0]), len(imgs[0][0])), dtype=np.float64)

    for f in range(no_images):
        for i in range(len(imgs[0])):
            for j in range(len(imgs[0][0])):
                if C[f][i][j] != 0:
                    weightMap[f][i][j] = C[f][i][j] * S[f][i][j] * E[f][i][j]
                else:
                    weightMap[f][i][j] = S[f][i][j] * E[f][i][j]

    weightMap = normalizeWP(weightMap)

    return weightMap