import numpy as np
import cv2

"""
Many of the functions on this file are identical to the ones submitted for assignments 4 (blending).
One of the functions that is modified intensely is the blend function which now can blend not only 
two but many pyramids.  
"""


def normalize(img):
    """This function is identical to the one submitted for the
    Blending assignment.
    """

    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def viz_pyramid(stack, shape, name, norm=False):
    """This function is identical to the one submitted for the
    Blending assignment.
    """

    layers = [normalize(np.dstack(imgs)) if norm else np.clip(np.dstack(imgs), 0, 255) for imgs in zip(*stack)]
    stack = [cv2.resize(layer, shape, interpolation=3) for layer in layers]
    img = np.vstack(stack).astype(np.uint8)
    cv2.imwrite(name + ".png", img)
    return img


def generatingKernel(a):
    """This function is identical to the one submitted for the
    Blending assignment.
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """This function is identical to the one submitted for the
    Blending assignment.
    """

    # WRITE YOUR CODE HERE.
    #padding
    N, W = kernel.shape

    image = image.astype(np.float64)

    paddedImg = cv2.copyMakeBorder(image, (N-1)//2, (N-1)//2, (N-1)//2, (N-1)//2, borderType = cv2.BORDER_REFLECT_101)

    #Convolution
    filteredImg = cv2.filter2D(paddedImg, -1, kernel)

    #resizing
    resizedFilteredImg = filteredImg[(N-1)//2:filteredImg.shape[0] - (N-1)//2, (N-1)//2:filteredImg.shape[1] - (N-1)//2]

    #subsampling
    subImg = resizedFilteredImg[::2, ::2]

    subImg = subImg.astype(np.float64)

    return subImg


def expand_layer(image, kernel=generatingKernel(0.4)):
    """This function is identical to the one submitted for the
    Blending assignment.
    """

    # WRITE YOUR CODE HERE.
    N , W = kernel.shape

    image = image.astype(np.float64)

    cntr = 1

    for j in range(len(image[0])):
        image = np.insert(image, cntr, 0, axis=1)
        cntr += 2

    j = 0
    cntr = 1

    for j in range(len(image)):
        image = np.insert(image, [cntr], np.zeros(len(image[0])), axis=0)
        cntr += 2


    # padding
    paddedImg = cv2.copyMakeBorder(image, (N-1)//2, (N-1)//2, (N-1)//2, (N-1)//2, borderType=cv2.BORDER_REFLECT_101)

    filteredImg = cv2.filter2D(paddedImg, -1, kernel)

    resizedFilteredImg = filteredImg[(N-1)//2:filteredImg.shape[0] - (N-1)//2, (N-1)//2:filteredImg.shape[1] - (N-1)//2]

    resizedFilteredImg *= 4

    resizedFilteredImg = resizedFilteredImg.astype(np.float64)

    return resizedFilteredImg


def gaussPyramid(image, levels):
    """This function is identical to the one submitted for the
    Blending assignment.
    """

    # WRITE YOUR CODE HERE.

    tempImg = image
    tempImg = tempImg.astype(np.float)
    images = []
    images.append(tempImg)

    for n in range(levels):
        tempImg = reduce_layer(tempImg)
        tempImg = tempImg.astype(np.float)
        images.append(tempImg)

    return images


def laplPyramid(gaussPyr):
    """This function is identical to the one submitted for the
    Blending assignment.
    """

    images = []

    for n in range(len(gaussPyr) - 1):
        expandedImg = expand_layer(gaussPyr[n+1])

        if (gaussPyr[n].shape == expandedImg.shape):
            tempImg = gaussPyr[n] - expandedImg
            tempImg = tempImg.astype(np.float)
            images.append(tempImg)
        else:
            if (gaussPyr[n].shape[1] < expandedImg.shape[1]):
                expandedImg = expandedImg[:, 0:expandedImg.shape[1] - 1]

            if (gaussPyr[n].shape[0] - expandedImg.shape[0]):
                expandedImg = expandedImg[0:expandedImg.shape[0] - 1, :]

            tempImg = gaussPyr[n] - expandedImg
            tempImg = tempImg.astype(np.float)
            images.append(tempImg)

    tempImg = gaussPyr[len(gaussPyr) - 1]
    tempImg = tempImg.astype(np.float)
    images.append(gaussPyr[len(gaussPyr) - 1])

    return images


def blend(img_list, weight_list):
    """This function is NOT similar to the one submitted for the
    Blending assignment.
    The blend function submitted for assignment no. 4 blends the layers
    of two pyramids which are the laplacian pyramids for the black and
    white pictures. This function is modified to be able to blend as many
    pyramids as the number of input images. This function blends all the
    pyramids according to their appropriate weight calculated in
    weight_list (weightMap.py).
    """

    images = []

    no_images = len(img_list)
    no_layers = len(img_list[0])

    for n in range(no_layers):

        image = np.zeros(img_list[0][n].shape)
        rows = len(img_list[0][n])
        columns = len(img_list[0][n][0])

        for i in range(rows):
            for j in range(columns):
                for k in range(no_images):
                    image[i][j] += img_list[k][n][i][j] * weight_list[k][n][i][j]

        image = image.astype(np.float)
        images.append(image)

    return images


def collapse(pyramid):
    """This function is identical to the one submitted for the
    Blending assignment.
    """

    image = np.zeros(pyramid[0].shape)
    index = len(pyramid) - 1
    image = pyramid[index]

    for n in range(len(pyramid) - 1):
        expandedImg = expand_layer(image)
        if (pyramid[index - 1].shape == expandedImg.shape):
            tempImg = pyramid[index - 1] + expandedImg
            tempImg = tempImg.astype(np.float)
            image = (tempImg)
        else:
            if (pyramid[index - 1].shape[1] < expandedImg.shape[1]):
                expandedImg = expandedImg[:, 0:expandedImg.shape[1] - 1]

            if (pyramid[index - 1].shape[0] < expandedImg.shape[0]):
                expandedImg = expandedImg[:expandedImg.shape[0] - 1, :]

            tempImg = pyramid[index - 1] + expandedImg
            tempImg = tempImg.astype(np.float)
            image = (tempImg)

        index -= 1

    return image
