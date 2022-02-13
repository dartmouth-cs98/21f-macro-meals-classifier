# for testing random new libraries and features

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, color, data
import cv2
from skimage.exposure import histogram
from skimage.feature import canny


import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

# image = data.imread(, as_grey = True)  # Load Image
image = cv2.imread("train_images/Apple/1_1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# image = data.binary_blobs()

# def image_show(image, nrows=1, ncols=1, cmap='gray'):
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
#     ax.imshow(image, cmap='gray')
#     ax.axis('off')
#     return fig, ax
#
# fig, ax = plt.subplots(1, 1)
# ax.hist(image.ravel(), bins=32, range=[0, 256])
# ax.set_xlim(0, 256);
#
# image_segmented = image>70
# image_show(image_segmented);
#
# # image_threshold = filters.threshold_otsu(image)
# # image_show(image>image_threshold)
#
# plt.show()

########################

# threshold = filters.threshold_otsu(image)  # Calculate threshold
# image_thresholded = image > threshold  # Apply threshold
image_thresholded = image > 50  # Apply threshold

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image, 'gray')
ax[1].imshow(image_thresholded, 'gray')
ax[0].set_title("Intensity")
ax[1].set_title("Thresholded")

# Apply 2 times erosion to get rid of background particles
n_erosion = 2
image_eroded = image_thresholded
for x in range(n_erosion):
    image_eroded = morphology.binary_erosion(image_eroded)

# Apply 14 times dilation to close holes
n_dilation = 14
image_dilated = image_eroded
for x in range(n_dilation):
    image_dilated = morphology.binary_dilation(image_dilated)

# Apply 4 times erosion to recover original size
n_erosion = 4
image_eroded_two = image_dilated
for x in range(n_erosion):
    image_eroded_two = morphology.binary_erosion(image_eroded_two)

cross = np.array([[0,0,0,0,0], [0,0,1,0,0], [0,1,1,1,0],
                [0,0,1,0,0], [0,0,0,0,0]], dtype=np.uint8)
cross_eroded = morphology.binary_erosion(cross)
cross_dilated = morphology.binary_dilation(cross)

labels = morphology.label(image_eroded_two)
labels_rgb = color.label2rgb(labels,
                             colors=['greenyellow', 'green',
                                     'yellow', 'yellowgreen'],
                             bg_label=0)
image.shape
# (342, 382)
labels.shape
# (342, 382)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(labels==1, 'gray')
ax[0,1].imshow(labels==2, 'gray')
ax[1,0].imshow(labels==3, 'gray')
ax[1,1].imshow(labels_rgb)
ax[0,0].set_title("label == 1")
ax[0,1].set_title("label == 2")
ax[1,0].set_title("label == 3")
ax[1,1].set_title("All labels RGB")

plt.show()


print("hello")
