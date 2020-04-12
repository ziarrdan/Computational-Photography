# Exposure Fusion
In this project exposure fusion technique is implemented for creating final artifacts that are capable of showing a high dynamic range of a scene. The final artifact of exposure fusion is live, beautiful and looks like an artistic creation. 

### Project Pipeline
The contrast, saturation and well-exposedness are calculated for each input image by laplacian, saturation and exposedness functions in weightMap.py. These are the three quality measures described in the paper by T. Mertens et al. and are used as a measure to decide how much weight should be assigned to each pixel. The weight map for each image is then calculated by multiplying all the measures for each pixel. Finally, the weight maps are normalized across the image stack which is done by normalizeWP function in weightMap.py. The laplacian pyramid of the input images are then calculated by the laplPyramid function in blending.py (this function is identical to the one submitted for the blending assignment). The calculated laplacian pyramid and the normalized weight maps are then used to blend the input images. The blended images are then collapsed to form the final output.

libraries for A1 | Version
--------------|------------
OpenCV | 4.2.0.34
numpy | 1.18.1
scipy | 1.4.1

## How to Run
To run the code, simply run the exposureFusion.py file under the main directory. The figures are generated under its corresponding  "output" folder under images. Make sure to copy all your images to "images/source".

## Results
Input Images:

|||
--------------|------------
|![First Input Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/source/IMG_0437.JPG) | ![Second Input Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/source/IMG_0436.JPG) |
|![Third Input Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/source/IMG_0435.JPG) | ![Fourth Input Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/source/IMG_0430.JPG) |
|![Fifth Input Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/source/IMG_0431.JPG) | ![Sixth Input Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/source/IMG_0432.JPG) |
|![Seventh Input Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/source/IMG_0433.JPG) | ![Eighth Input Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/source/IMG_0434.JPG) |

Output Image:

![Output Image](https://github.com/ziarrdan/CS6457-Computational-Photography/blob/master/exposureFusion/images/output/outimg.png)
